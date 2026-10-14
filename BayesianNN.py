import os
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
import pytorch_lightning as pl
import time
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances



# =============== 模型保存路径 ===============
start_time = time.time()
save_path = r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\new_dataset"

# 确保父目录存在
parent_dir = os.path.dirname(save_path)
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

if not os.path.exists(save_path):
    open(save_path, 'w').close()
    print(f"文件 '{save_path}' 创建成功")
else:
    print(f"文件 '{save_path}' 已经存在")

# =============== 数据模块 ===============
# 训练数据集
batch_size = 10
train_data = spk.data.AtomsDataModule(
    r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\dataset\Ta2Nn\Ta2N3-.db",  # 数据集1路径
    batch_size=batch_size,
    distance_unit='Ang',
    property_units={"energy_U0": 'eV'},
    num_train=0.9,
    num_val=0.1,
    split_file="dataset1_split.npz",
    transforms=[trn.ASENeighborList(cutoff=6.),
                trn.RemoveOffsets("energy_U0", remove_mean=True, remove_atomrefs=False),
                trn.CastTo32()],
    num_workers=1,
    pin_memory=True,
)

# 预测数据集
predict_data = spk.data.AtomsDataModule(
    # r"C:\Users\WHQ\Desktop\dataset\EuSin_coords\eusi7_array.txt",  # 数据集2路径
    datapath=r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\dataset\Ta2Nn\Ta2N5-.db",
    batch_size=10,
    distance_unit='Ang',
    property_units={"energy_U0": 'eV'},
    num_train=0.9,
    num_val=0.1,
    split_file="dataset2_split.npz",
    transforms=[trn.ASENeighborList(cutoff=6.),
                trn.RemoveOffsets("energy_U0", remove_mean=True, remove_atomrefs=False),
                trn.CastTo32()],
    num_workers=1,
    pin_memory=True,
)

train_data.prepare_data()
predict_data.prepare_data()

if __name__ == '__main__':
    # 旧csv
    savepath = r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\ta2n3-5\ta2n3-5.csv"
    # 四种结构
    save_str_dir = r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\ta2n3-5"
    # 新数据集
    third_dataset_path = os.path.join(save_path, "ta2n3-5.db")
    # 可视化 新csv
    new_csv_path = r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\ta2n3-5\ta2n3-5_tsne.csv"
    start_time = time.time()

    # 训练模型
    train_data.setup()
    print(f"训练集大小: {len(train_data.train_dataset)}")
    print(f"验证集大小: {len(train_data.val_dataset)}")

    # =============== 模型超参数 ===============
    cutoff = 6.
    n_atom_basis = 16
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=1,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    pred_U0 = spk.atomistic.BayesianNN(n_in=n_atom_basis, output_key="energy_U0")

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=pred_U0,
        postprocessors=[trn.CastTo64(),
                        trn.AddOffsets("energy_U0", add_mean=True, add_atomrefs=False)]
    )

    output_U0 = spk.task.ModelOutput_BNN(
        name="energy_U0",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={"MAE": torchmetrics.MeanAbsoluteError(),
                 "MSE": torchmetrics.MeanSquaredError()
                 },
        # main_output_key="energy_U0",
        # additional_output_keys=["uncertainty"],
    )

    task = spk.task.AtomisticTask_BNN(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3}
    )

    # =============== 日志与回调 ===============
    logger = pl.loggers.CSVLogger(
        save_dir=r"C:\Users\WHQ\Desktop\Active_Learning_2025.10\ta2n3-5",
        name='ta2n3-5'
    )
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(save_path, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss",
        )
    ]

    # =============== 训练 ===============
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=save_path,
        max_epochs=100
        ,
    )
    trainer.fit(task, datamodule=train_data)

    # =============== 预测第二个数据集 ===============
    print("\n开始预测第二个数据集的能量并计算不确定性...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_data.setup()

    # 使用正确的模型加载方法

    from schnetpack.utils import load_model
    model = load_model(
        os.path.join(save_path, "best_inference_model"),
        device=device
    )
    model.eval()
    model.output_modules.reset_tsne_cache()

    # 初始化转换器
    import schnetpack as spk
    from schnetpack import interfaces

    converter = interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        dtype=torch.float32
    )

    # 获取预测数据集的预测能量和不确定性
    all_uncertainties = []
    all_energies = []
    all_structures = []
    # all_atoms_objects = []

    for batch in predict_data.train_dataloader():
        batch = {key: value.to(device) if hasattr(value, 'to') else value
                 for key, value in batch.items()}

        # 获取预测结果
        with torch.no_grad():
            outputs = model(batch)



        # 提取预测能量和不确定性
        energies = outputs["energy_U0"].cpu().detach().numpy()
        all_energies.append(energies)
        uncertainties = None
        # 优先从模型缓存获取不确定性（BNN专属，更准确）
        uncertainty_dict = model.output_modules.get_cached_uncertainty()
        uncertainties = uncertainty_dict["std"].cpu().detach().numpy()
        all_uncertainties.append(uncertainties)

        # 保存原始结构信息用于后续操作
        if "_idx" in batch and "_positions" in batch:
            arr_n_atoms = batch["_n_atoms"].cpu().detach().numpy()
            n_atoms = arr_n_atoms[0]
            batch_structures = batch["_positions"].cpu().detach().numpy()
            normal_structures = batch_structures.reshape(batch_size, n_atoms, 3)
            all_structures.append(normal_structures)
            # all_atoms_objects.append(batch)

    all_energies = np.concatenate(all_energies, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_structures = np.concatenate(all_structures, axis=0)
    model.output_modules.export_tsne_csv(new_csv_path)
    # 找到相应的索引
    idx_lowest_energy = np.argmin(all_energies)
    idx_highest_energy = np.argmax(all_energies)
    idx_highest_uncertainty = np.argmax(all_uncertainties)
    idx_lowest_uncertainty = np.argmin(all_uncertainties)

    # 创建保存目录
    import os


    # 构建DataFrame
    data = pd.DataFrame({
        "Energy": all_energies.flatten(),
        "Uncertainty": all_uncertainties.flatten()
    })

    # 保存为CSV
    data.to_csv(savepath, index=False)


    os.makedirs(save_str_dir, exist_ok=True)


    # 定义保存函数（根据你的实际需求调整格式，比如XYZ、PDB等）
    def save_structure(structure, energy, uncertainty, filename):
        """
        保存结构信息到文件
        structure: (n_atoms, 3) 的坐标数组
        energy: 能量值
        uncertainty: 不确定性值
        """
        with open(os.path.join(save_str_dir, filename), 'w') as f:
            n_atoms = structure.shape[0]
            f.write(f"{n_atoms}\n")
            f.write(f"Energy: {energy:.6f}, Uncertainty: {uncertainty:.6f}\n")
            for i, coord in enumerate(structure):
                f.write(f"X {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


    # 保存四种结构
    save_structure(all_structures[idx_lowest_energy],
                   all_energies[idx_lowest_energy],
                   all_uncertainties[idx_lowest_energy],
                   "lowest_energy.xyz")

    save_structure(all_structures[idx_highest_energy],
                   all_energies[idx_highest_energy],
                   all_uncertainties[idx_highest_energy],
                   "highest_energy.xyz")

    save_structure(all_structures[idx_highest_uncertainty],
                   all_energies[idx_highest_uncertainty],
                   all_uncertainties[idx_highest_uncertainty],
                   "highest_uncertainty.xyz")

    save_structure(all_structures[idx_lowest_uncertainty],
                   all_energies[idx_lowest_uncertainty],
                   all_uncertainties[idx_lowest_uncertainty],
                   "lowest_uncertainty.xyz")

    # 可选：打印统计信息
    print(f"能量最低: {all_energies[idx_lowest_energy]:.6f} (不确定性: {all_uncertainties[idx_lowest_energy]:.6f})")
    print(f"能量最高: {all_energies[idx_highest_energy]:.6f} (不确定性: {all_uncertainties[idx_highest_energy]:.6f})")
    print(
        f"不确定性最高: {all_uncertainties[idx_highest_uncertainty]:.6f} (能量: {all_energies[idx_highest_uncertainty]:.6f})")
    print(
        f"不确定性最低: {all_uncertainties[idx_lowest_uncertainty]:.6f} (能量: {all_energies[idx_lowest_uncertainty]:.6f})")

    # =============== 计算不确定性高的结构 ===============
    uncertainty_threshold = np.percentile(all_uncertainties, 90)
    high_uncertainty_indices = np.where(all_uncertainties > uncertainty_threshold)[0]

    print(f"找到 {len(high_uncertainty_indices)} 个高不确定性结构")

    # =============== 计算相似性 ===============
    from sklearn.metrics.pairwise import pairwise_distances
    '''*******************************'''
    high_uncertainty_structures = all_structures[high_uncertainty_indices]
    high_uncertainty_energies = all_energies[high_uncertainty_indices]

    # 计算结构之间的欧式距离
    distances = pairwise_distances(
        high_uncertainty_structures.reshape(len(high_uncertainty_structures), -1),
        metric='euclidean'
    )

    # 对于每个结构，计算与其他结构的平均距离
    mean_distances = np.mean(distances, axis=1)

    # 筛选距离最远的结构（相似性最低）
    similarity_threshold = np.percentile(mean_distances, 80)
    diverse_indices = np.where(mean_distances > similarity_threshold)[0]
    '''************************************'''
    selected_structures = high_uncertainty_structures[diverse_indices]
    selected_energies = high_uncertainty_energies[diverse_indices]
    '''
    numbers = all_atoms_objects[0]['_atomic_numbers'][:8].cpu().tolist()
    '''
    print(f"筛选出 {len(diverse_indices)} 个多样化且高不确定性的结构")

    # =============== 保存第三个数据集 ===============
    # 使用SchNetPack的标准数据库格式保存


    # 清空旧数据库（如果存在）
    if os.path.exists(third_dataset_path):
        os.remove(third_dataset_path)

    # 保存选中的结构到数据库
    from ase.db import connect
    from ase import Atoms

    atoms_list = []
    property_list = []

    # 遍历已经筛选好的 selected_structures（它和 selected_energies 对应）
    for i, positions in enumerate(selected_structures):
        positions = np.asarray(positions)  # shape = (N, 3)
        N = positions.shape[0]

        # 构造 numbers：假设第一个原子是 Eu (Z=63)，其余是 Si (Z=14)
        # 如果你有其它元素或顺序，请在这里调整
        numbers = np.array([73] * 2 + [7] * (N - 2), dtype=int)

        ats = Atoms(numbers=numbers, positions=positions)

        props = {"energy_U0": selected_energies[i]}
        atoms_list.append(ats)
        property_list.append(props)

    # 然后写入 ASE/SchNetPack 数据库（你的原逻辑）
    from schnetpack.data import ASEAtomsData
    #

    new_dataset = ASEAtomsData.create(
        third_dataset_path,
        distance_unit="Ang",
        property_unit_dict={"energy_U0": "eV"},
    )
    new_dataset.add_systems(property_list, atoms_list)

    print(f"已将 {len(diverse_indices)} 个结构保存到 {third_dataset_path}")

    # =============== 结束计时 ===============
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n完成预测和数据筛选，总耗时: {duration / 60:.2f} 分钟")
