import os
import math
import torch
import numpy as np

def load_tensor(path, delim=','):
    arr = np.loadtxt(path, delimiter=delim)
    return torch.from_numpy(arr).float()

def load_sensor_parameters(para_dir, device):
    params = []
    for i in range(16):
        p = np.loadtxt(
            os.path.join(para_dir, f"para_sensor_16_{i+1}.txt"),
            delimiter=','
        )
        params.append(p)
    return torch.tensor(params, dtype=torch.float32, device=device)


def orientation_from_theta_phi(theta, phi):
    """
    Rule-based dipole orientation (discrete)
    """
    if theta == 90 and phi == 0:
        return 1, 0, 0
    if theta == -90 and phi == 0:
        return -1, 0, 0
    if theta == 0 and phi == 90:
        return 0, 1, 0
    if theta == 0 and phi == -90:
        return 0, -1, 0
    if theta == 0 and phi == 0:
        return 0, 0, 1
    if theta == 0 and phi == 180:
        return 0, 0, -1
    raise ValueError(f"Undefined orientation: theta={theta}, phi={phi}")


def generate_dipole_data_torch(cc, sensor_params):
    """
    cc: (N, 5) -> [x,y,z,theta,phi]
    return: (N, 48)
    """
    cc = cc.float()
    N = cc.shape[0]
    device = cc.device

    xl, yl, zl = cc[:, 0], cc[:, 1], cc[:, 2]
    theta, phi = cc[:, 3], cc[:, 4]

    m = torch.zeros(N, device=device)
    n = torch.zeros(N, device=device)
    p = torch.zeros(N, device=device)

    for i in range(N):
        mi, ni, pi = orientation_from_theta_phi(
            theta[i].item(), phi[i].item()
        )
        m[i], n[i], p[i] = mi, ni, pi

    B_all = torch.zeros((N, 48), device=device)

    for s in range(16):
        BT = sensor_params[s, 0]
        a, b, c = sensor_params[s, 1:4]
        ox, oy, oz = sensor_params[s, 4:7]
        roll, pitch, yaw = sensor_params[s, 7:10]

        rx, ry, rz = xl - a, yl - b, zl - c
        R = torch.sqrt(rx**2 + ry**2 + rz**2)
        mdotr = m*rx + n*ry + p*rz

        Blx = -BT * (3*mdotr*rx/R**5 - m/R**3) * 1e6 + ox
        Bly = -BT * (3*mdotr*ry/R**5 - n/R**3) * 1e6 + oy
        Blz = -BT * (3*mdotr*rz/R**5 - p/R**3) * 1e6 + oz

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        Rz = torch.tensor([[cy,-sy,0],[sy,cy,0],[0,0,1]], device=device)
        Ry = torch.tensor([[cp,0,sp],[0,1,0],[-sp,0,cp]], device=device)
        Rx = torch.tensor([[1,0,0],[0,cr,-sr],[0,sr,cr]], device=device)
        Rm = Rz @ Ry @ Rx

        B = torch.stack([Blx, Bly, Blz], dim=0)
        B_all[:, s*3:(s+1)*3] = (Rm @ B).T

    return B_all


def generate_dipole_data(cc_file, para_dir, N_sensors=16):
    """
    根据校准参数生成磁偶极子模拟数据 (NumPy版)
    cc_file: 轨迹文件 (N,6) -> [x, y, z, theta, phi, psi]
    para_dir: 16个传感器参数路径
    N_sensors: 传感器数量
    返回:
        B_all: (N, N_sensors*3)
    """
    cc = np.loadtxt(cc_file, delimiter=",")
    xl, yl, zl = cc[:, 0] * 1e-3, cc[:, 1] * 1e-3, cc[:, 2] * 1e-3
    theta, phi = cc[:, 3], cc[:, 4]
    N_points = xl.shape[0]

    # 生成 m,n,p
    m = np.zeros(N_points)
    n = np.zeros(N_points)
    p = np.zeros(N_points)
    for i in range(N_points):
        th, ph = theta[i], phi[i]
        if th == 90 and ph == 0:
            m[i], n[i], p[i] = 1, 0, 0
        elif th == -90 and ph == 0:
            m[i], n[i], p[i] = -1, 0, 0
        elif th == 0 and ph == 90:
            m[i], n[i], p[i] = 0, 1, 0
        elif th == 0 and ph == -90:
            m[i], n[i], p[i] = 0, -1, 0
        elif th == 0 and ph == 0:
            m[i], n[i], p[i] = 0, 0, 1
        elif th == 0 and ph == 180:
            m[i], n[i], p[i] = 0, 0, -1
        else:
            raise ValueError(f"未定义角度: theta={th}, phi={ph}")

    B_all = np.zeros((N_points, N_sensors * 3))
    for i in range(N_sensors):
        para_file = os.path.join(para_dir, f"para_sensor_16_{i + 1}.txt")
        BT, a, b, c, ox, oy, oz, roll, pitch, yaw = np.loadtxt(para_file)
        rx, ry, rz = xl - a, yl - b, zl - c
        R = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        mdotr = m * rx + n * ry + p * rz
        Blx = -BT * (3 * mdotr * rx / R ** 5 - m / R ** 3) * 1e6 + ox
        Bly = -BT * (3 * mdotr * ry / R ** 5 - n / R ** 3) * 1e6 + oy
        Blz = -BT * (3 * mdotr * rz / R ** 5 - p / R ** 3) * 1e6 + oz
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_mat = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ \
                np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ \
                np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        B_rot = (R_mat @ np.vstack([Blx, Bly, Blz])).T
        B_all[:, i * 3:(i + 1) * 3] = B_rot
    return B_all

