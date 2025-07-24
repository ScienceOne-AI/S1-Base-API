import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator


def geometry_visualization(points: np.ndarray) -> np.ndarray:
    """
    生成点云构型可视化，保存到 results 目录，文件名带时间戳后缀，
    并返回图像的 NumPy 数组 (H, W, 3)。
    """
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'results/geometry_vis_{timestamp}.png'

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c='blue')
    ax.set_title("3D Geometry Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # 获取数据范围
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    # 计算范围并添加10%的边距
    x_range = (x_max - x_min) * 1.1 or 1.0
    y_range = (y_max - y_min) * 1.1 or 1.0
    z_range = (z_max - z_min) * 1.1 or 1.0

    # 设置坐标轴范围
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)

    # 关键修改：设置X轴拉伸5倍
    # 使用5倍比例因子实现X轴拉伸
    stretch_factor = 1.
    ax.set_box_aspect([stretch_factor * x_range, y_range, z_range])
    ax.yaxis.set_major_locator(MaxNLocator(3))  # Y轴最多显示3个刻度
    ax.zaxis.set_major_locator(MaxNLocator(3))  # Z轴最多显示3个刻度

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    # 使用 Agg 画布渲染图像
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # 获取渲染后的图像数据
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img = img[..., :3]  # 去掉 alpha 通道，只保留 RGB

    plt.close(fig)

    print(f"Saved point cloud geometry visualization to {output_path}")
    return img


def pressure_visualization(points: np.ndarray) -> np.ndarray:
    """
    生成压力场可视化，保存到 results 目录，文件名带时间戳后缀，
    并返回图像的 NumPy 数组 (H, W, 3)。
    """
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'results/pressure_vis_{timestamp}.png'

    x, y, z, attr = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=attr, cmap='jet', s=0.5, alpha=1.0)
    ax.set_title("Surface Pressure Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # 获取数据范围
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    # 计算范围并添加10%的边距
    x_range = (x_max - x_min) * 1.1 or 1.0
    y_range = (y_max - y_min) * 1.1 or 1.0
    z_range = (z_max - z_min) * 1.1 or 1.0

    # 设置坐标轴范围
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)

    # 关键修改：设置X轴拉伸5倍
    # 使用5倍比例因子实现X轴拉伸
    stretch_factor = 1.
    ax.set_box_aspect([stretch_factor * x_range, y_range, z_range])
    ax.yaxis.set_major_locator(MaxNLocator(3))  # Y轴最多显示3个刻度
    ax.zaxis.set_major_locator(MaxNLocator(3))  # Z轴最多显示3个刻度

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Pressure Coefficient")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    # 使用 Agg 画布渲染图像
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # 获取渲染后的图像数据
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img = img[..., :3]  # 去掉 alpha 通道，只保留 RGB

    plt.close(fig)

    print(f"Saved pressure field visualization to: {output_path}")
    return img
