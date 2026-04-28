import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import threading
import tkinter as tk
from tkinter import ttk

# 设置中文字体，防止Matplotlib绘图时中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import os
from dotenv import load_dotenv

load_dotenv()


class AgentParser:
    """大模型智能体解析器类"""

    def __init__(self):
        from openai import OpenAI
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError("未找到 API Key，请检查 .env 文件是否配置正确！")

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        self.system_prompt = """
        你是一个 Delta 机器人的高级控制大脑与矢量图形引擎。请将用户的指令解析为严格的 JSON。

        Z轴规范：安全高度Z=-300（用于抬笔移动），工作平面Z=-400（用于落笔画线）。

        返回格式（action 为 path 用于画图）：
        {"action": "path", "waypoints": [[x1, y1, z1, p1], [x2, y2, z2, p2], ...]} 
           - 此模式用于画图、画线。
           - 第四个参数 p 代表画笔状态。p=1 表示落笔画线，p=0 表示抬笔移动（不留痕迹）。
           - 抬笔与落笔逻辑示例：要画两条平行线，需先 p=0 移至线1起点，再 p=1 画至线1终点，然后 p=0 移至线2起点，p=1 画至线2终点。

        【核心绘图法则：应对具象图形（如动物、人脸）】
        由于你缺乏视觉反馈，在绘制复杂五官时，必须严格遵循“几何拆解”与“绝对对称”法则：
        1. 轮廓近似：用正多边形（如八边形、十二边形）近似圆。
        2. 绝对对称：五官（眼睛、耳朵、鼻孔）必须严格关于 Y 轴左右对称！例如左眼中心在 X=-20，右眼中心必须在 X=20。
        3. 抬笔跨越：画完左眼后，必须用 p=0 移动到右眼起点，绝不能一笔连成。

        【高级样例1：画一个边长80的正方体】
        用户指令：“画一个边长80的正方体”
        逻辑拆解：底面Z=-400，顶面Z=-320。
        示例 JSON：
        ```json
        {"action": "path", "waypoints": [
          [40,40,-400,0], [40,-40,-400,1], [-40,-40,-400,1], [-40,40,-400,1], [40,40,-400,1], 
          [40,40,-320,0], [40,-40,-320,1], [-40,-40,-320,1], [-40,40,-320,1], [40,40,-320,1], 
          [40,40,-400,0], [40,40,-320,1], 
          [40,-40,-400,0], [40,-40,-320,1], 
          [-40,-40,-400,0], [-40,-40,-320,1], 
          [-40,40,-400,0], [-40,40,-320,1] 
        ]}
        ```
        【高级样例2：画一只猪】
        用户指令：“画一只猪”
        逻辑拆解：绘制一个正脸卡通猪头。拆解为：八边形大脸、左右对称的三角形耳朵、左右对称的两段眼线、六边形猪鼻子、鼻孔、V型微笑。
        坐标在 X:[-100, 100], Y:[-100, 150] 范围内设定。
        示例 JSON：
        ```json
        {"action": "path", "waypoints": [
          [30,80,-400,0], [60,50,-400,1], [80,0,-400,1], [60,-50,-400,1], [30,-80,-400,1], [-30,-80,-400,1], [-60,-50,-400,1], [-80,0,-400,1], [-60,50,-400,1], [-30,80,-400,1], [30,80,-400,1], // 八边形脸
          [-30,80,-400,0], [-60,120,-400,1], [-60,50,-400,1], // 左耳朵
          [30,80,-400,0], [60,120,-400,1], [60,50,-400,1], // 右耳朵
          [-35,20,-400,0], [-15,20,-400,1], // 左眼
          [35,20,-400,0], [15,20,-400,1], // 右眼
          [-20,-10,-400,0], [20,-10,-400,1], [30,-30,-400,1], [20,-50,-400,1], [-20,-50,-400,1], [-30,-30,-400,1], [-20,-10,-400,1], // 六边形猪鼻
          [-10,-30,-400,0], [-10,-40,-400,1], // 左鼻孔
          [10,-30,-400,0], [10,-40,-400,1], // 右鼻孔
          [-20,-65,-400,0], [0,-75,-400,1], [20,-65,-400,1] // 微笑嘴巴
        ]}
        ```
        【行动指令】
        1. {"action": "path", "waypoints": [[x1, y1, z1, p1], ...]} 用于绘图。
        2. 画复杂图形时，严格应用 p=0 抬笔跨越和对称法则。
        3. 只返回严格的 JSON 对象。
        """

    def parse(self, text, current_pos):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": text}],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=50.0
            )
            print(json.loads(response.choices[0].message.content))
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"API 请求失败，启用本地降级匹配: {str(e)}")
            return self.local_fallback_parse(text)

    def local_fallback_parse(self, text):
        text = text.lower()
        if "圆" in text: return {"action": "circle"}
        if "三角" in text: return {"action": "triangle"}
        if "星" in text: return {"action": "star"}
        if "抓" in text or "传送" in text: return {"action": "pick_and_place"}
        if "复位" in text or "原点" in text: return {"action": "home"}
        return {"error": "无法理解指令"}


class DeltaRobot:
    """Delta并联机器人运动学核心类"""

    def __init__(self, r_base=150, r_end=50, l_bicep=200, l_forearm=450, w_rod=40):
        self.R, self.r, self.L1, self.L2, self.W = r_base, r_end, l_bicep, l_forearm, w_rod
        self.angles = np.radians([0, 120, 240])

    def inverse_kinematics(self, x, y, z):
        thetas = []
        for i in range(3):
            alpha = self.angles[i]
            x_local = x * np.cos(alpha) + y * np.sin(alpha)
            y_local = -x * np.sin(alpha) + y * np.cos(alpha)
            p_x, p_y, p_z = x_local + self.r - self.R, y_local, z

            E1, F1 = 2 * self.L1 * p_x, -2 * self.L1 * p_z
            G1 = p_x ** 2 + p_y ** 2 + p_z ** 2 + self.L1 ** 2 - self.L2 ** 2
            D = F1 ** 2 - G1 ** 2 + E1 ** 2

            if D < 0:
                if D > -1e-6:
                    D = 0.0
                else:
                    return None

            denom = G1 + E1
            if abs(denom) < 1e-8:
                return None

            t = (F1 - np.sqrt(D)) / denom
            thetas.append(2 * np.arctan(t))
        return thetas


class PhysicalPID:
    """
    改进的物理PID控制器：
    增加了抗积分饱和(Anti-windup)、加速度限幅与速度限幅，提高系统跟飞目标的稳定性。
    """

    def __init__(self):
        self.pos = np.array([0.0, 0.0, -300.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        # 优化默认PID参数：降低Kp防止过冲，增加Kd提高阻尼感
        self.kp, self.ki, self.kd = 0.10, 0.001, 0.5

    def update(self, target_pos, dt=1.0):
        target = np.array(target_pos)
        error = target - self.pos

        # 1. 积分项更新与抗积分饱和 (Anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -50.0, 50.0)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        # 2. 计算理想加速度
        acceleration = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 3. 加速度限幅：防止瞬间极大的位置误差导致系统崩溃或疯狂震荡
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > 8.0:
            acceleration = (acceleration / acc_norm) * 8.0

        # 4. 速度更新与强阻尼衰减
        self.vel += acceleration * dt
        self.vel *= 0.85  # 【关键修改】从 0.95 降至 0.85，增大物理环境摩擦力，极大减少震荡

        # 5. 速度限幅：保证末端执行器不会移动过快
        vel_norm = np.linalg.norm(self.vel)
        if vel_norm > 6.0:
            self.vel = (self.vel / vel_norm) * 6.0

        # 6. 位置更新
        self.pos += self.vel * dt
        return self.pos, np.linalg.norm(self.vel), np.linalg.norm(acceleration)


# 实例化核心组件
robot = DeltaRobot()
agent = AgentParser()
pid_sys = PhysicalPID()

# 全局状态变量
current_frame, total_frames = 0, 100
traj_x, traj_y, traj_z = np.full(100, 0.0), np.full(100, 0.0), np.full(100, -300.0)
traj_g = np.full(100, False)
traj_p = np.full(100, 1)

target_th_profile, target_dth_profile, target_ddth_profile = [], [], []
actual_pos_h = {"x": [], "y": [], "z": []}
draw_h = {"x": [], "y": [], "z": []}
actual_th_h, actual_dth_h, actual_ddth_h = [], [], []
target_th_h = []

use_s_curve = True
show_obstacle = True
show_env_mode4 = False
is_processing = False
is_paused = False
clear_history_frame = -1
stop_draw_frame = -1
steady_counter = 0
current_mode = 0

obs_center = np.array([80, 80, -300])
obs_r = 50


def interpolate_velocity(start, end, steps):
    steps = max(2, int(steps))
    t = np.linspace(0, 1, steps)
    s = (3 * t ** 2 - 2 * t ** 3) if use_s_curve else t
    return start + (end - start) * s


def calc_joint_profiles(x_arr, y_arr, z_arr):
    th_arr = []
    for x, y, z in zip(x_arr, y_arr, z_arr):
        ths = robot.inverse_kinematics(x, y, z)
        if ths is not None:
            th_arr.append(np.degrees(ths[0]))
        else:
            th_arr.append(th_arr[-1] if th_arr else 0)
    th_arr = np.array(th_arr)

    if len(th_arr) > 1:
        dth_arr = np.pad(np.diff(th_arr), (0, 1), 'edge')
        ddth_arr = np.pad(np.diff(dth_arr), (0, 1), 'edge')
    else:
        dth_arr = np.zeros_like(th_arr)
        ddth_arr = np.zeros_like(th_arr)
    return th_arr, dth_arr, ddth_arr


def plan_trajectory(start_pt, end_pt):
    # 【关键修改】降低规划层的最大速度，让目标点移动更平缓，方便底层PID完美追踪
    v_max = 3.5  # 原来是 7.0
    start_arr = np.array(start_pt)
    end_arr = np.array(end_pt)
    dist = np.linalg.norm(end_arr - start_arr)

    frames = max(12, int(dist / v_max))

    if show_obstacle and not show_env_mode4 and dist > 1e-3:
        safe_r = obs_r + robot.r + 60
        d = end_arr[:2] - start_arr[:2]
        f = start_arr[:2] - obs_center[:2]

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - safe_r ** 2

        if a > 1e-6:
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)

                if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1):
                    t_closest = np.clip(-b / (2 * a), 0, 1)
                    closest_pt_xy = start_arr[:2] + t_closest * d
                    push_vec = closest_pt_xy - obs_center[:2]
                    push_dist = np.linalg.norm(push_vec)

                    if push_dist < 1e-3:
                        push_vec = np.array([-d[1], d[0]])
                        push_dist = np.linalg.norm(push_vec)

                    via_xy = obs_center[:2] + (push_vec / push_dist) * safe_r
                    via_z = max(start_arr[2], end_arr[2]) + 30
                    via_z = np.clip(via_z, -550, -280)
                    via_pt = np.array([via_xy[0], via_xy[1], via_z])

                    frames_half = max(2, frames // 2)

                    tx1 = interpolate_velocity(start_pt[0], via_pt[0], frames_half)
                    ty1 = interpolate_velocity(start_pt[1], via_pt[1], frames_half)
                    tz1 = interpolate_velocity(start_pt[2], via_pt[2], frames_half)

                    tx2 = interpolate_velocity(via_pt[0], end_pt[0], frames_half)
                    ty2 = interpolate_velocity(via_pt[1], end_pt[1], frames_half)
                    tz2 = interpolate_velocity(via_pt[2], end_pt[2], frames_half)

                    return np.concatenate([tx1, tx2]), np.concatenate([ty1, ty2]), np.concatenate([tz1, tz2])

    tx = interpolate_velocity(start_pt[0], end_pt[0], frames)
    ty = interpolate_velocity(start_pt[1], end_pt[1], frames)
    tz = interpolate_velocity(start_pt[2], end_pt[2], frames)
    return tx, ty, tz


def execute_path(waypoints, action_desc="移动", hide_approach=False, gripper_states=None, is_mode4=False):
    global traj_x, traj_y, traj_z, traj_g, traj_p, total_frames, current_frame, is_processing, is_paused
    global target_th_profile, target_dth_profile, target_ddth_profile
    global clear_history_frame, stop_draw_frame, steady_counter

    if not is_mode4:
        belt_line.set_visible(False)
        box_line.set_visible(False)
        workpiece.set_visible(False)
        global show_env_mode4
        show_env_mode4 = False

    update_status(f"状态: 运行中\n动作: {action_desc}", "blue")

    all_tx, all_ty, all_tz, all_g, all_p = [], [], [], [], []
    clear_idx = -1
    stop_draw_idx = -1

    for i in range(len(waypoints) - 1):
        pt1 = waypoints[i]
        pt2 = waypoints[i + 1]

        tx, ty, tz = plan_trajectory(pt1[:3], pt2[:3])
        pen_down = pt2[3] if len(pt2) > 3 else 1

        if hide_approach and i == 0 and len(waypoints) > 2:
            clear_idx = len(all_tx) + len(tx) - 1
            pen_down = 0

        if current_mode == 3 and i > 0 and np.linalg.norm(np.array(pt2[:3]) - np.array([0, 0, -300])) < 1.0:
            if stop_draw_idx == -1:
                stop_draw_idx = len(all_tx)

        gs = gripper_states[i] if (gripper_states and i < len(gripper_states)) else False

        all_tx.extend(tx)
        all_ty.extend(ty)
        all_tz.extend(tz)
        all_g.extend([gs] * len(tx))
        all_p.extend([pen_down] * len(tx))

    if not all_tx: return

    traj_x, traj_y, traj_z = np.array(all_tx), np.array(all_ty), np.array(all_tz)
    traj_g = np.array(all_g)
    traj_p = np.array(all_p)
    total_frames = len(traj_x)

    target_th_profile, target_dth_profile, target_ddth_profile = calc_joint_profiles(traj_x, traj_y, traj_z)

    clear_history_frame = clear_idx
    stop_draw_frame = stop_draw_idx
    steady_counter = 0

    actual_pos_h["x"].clear()
    actual_pos_h["y"].clear()
    actual_pos_h["z"].clear()
    draw_h["x"].clear()
    draw_h["y"].clear()
    draw_h["z"].clear()
    actual_th_h.clear()
    actual_dth_h.clear()
    actual_ddth_h.clear()
    target_th_h.clear()

    current_frame = 0
    is_processing = False
    is_paused = False


def draw_shape(shape_type):
    global current_mode
    current_mode = 2
    curr = [traj_x[-1], traj_y[-1], traj_z[-1]]
    z_plane = -400
    radius = 80

    if shape_type == 'circle':
        theta = np.linspace(0, 2 * np.pi, 50)  # 提升圆形的平滑度
        shape_pts = [[radius * np.cos(t), radius * np.sin(t), z_plane, 1] for t in theta]
        name = "圆形"
    elif shape_type == 'triangle':
        angles = np.radians([90, 210, 330, 90])
        shape_pts = [[radius * np.cos(a), radius * np.sin(a), z_plane, 1] for a in angles]
        name = "三角形"
    elif shape_type == 'star':
        angles = np.radians([90, 234, 18, 162, 306, 90])
        shape_pts = [[radius * np.cos(a), radius * np.sin(a), z_plane, 1] for a in angles]
        name = "五角星"
    else:
        return

    waypoints = [[curr[0], curr[1], curr[2], 0], [shape_pts[0][0], shape_pts[0][1], shape_pts[0][2], 0]] + shape_pts
    execute_path(waypoints, f"绘制 {name}", hide_approach=True)


def demo_conveyor():
    global show_env_mode4, current_mode
    current_mode = 4
    show_env_mode4 = True

    belt_w = 80
    pick_x, pick_y = 150, 100
    drop_x, drop_y = -150, -100

    bx = [pick_x - belt_w / 2, pick_x + belt_w / 2, pick_x + belt_w / 2, pick_x - belt_w / 2, pick_x - belt_w / 2]
    by = [pick_y - belt_w / 2, pick_y - belt_w / 2, pick_y + belt_w / 2, pick_y + belt_w / 2, pick_y - belt_w / 2]
    belt_line.set_data(bx, by)
    belt_line.set_3d_properties([-480] * 5)
    belt_line.set_visible(True)

    box_x = [drop_x - belt_w / 2, drop_x + belt_w / 2, drop_x + belt_w / 2, drop_x - belt_w / 2, drop_x - belt_w / 2]
    box_y = [drop_y - belt_w / 2, drop_y - belt_w / 2, drop_y + belt_w / 2, drop_y + belt_w / 2, drop_y - belt_w / 2]
    box_line.set_data(box_x, box_y)
    box_line.set_3d_properties([-450] * 5)
    box_line.set_visible(True)

    workpiece.set_data([pick_x], [pick_y])
    workpiece.set_3d_properties([-480])
    workpiece.set_visible(True)

    curr = [traj_x[-1], traj_y[-1], traj_z[-1]]
    belt_hover, belt_pick = [pick_x, pick_y, -300], [pick_x, pick_y, -480]
    box_hover, box_drop = [drop_x, drop_y, -300], [drop_x, drop_y, -450]

    waypoints = [curr, belt_hover, belt_pick, belt_hover, box_hover, box_drop, box_hover, [0, 0, -300]]
    gripper_states = [False, False, True, True, True, False, False]

    execute_path(waypoints, f"抓取 ({pick_x},{pick_y}) -> ({drop_x},{drop_y})",
                 gripper_states=gripper_states, is_mode4=True)


def execute_action(action_dict):
    curr_pos = [traj_x[-1], traj_y[-1], traj_z[-1]]
    action = action_dict.get('action', 'home')

    if action in ['circle', 'triangle', 'star']:
        draw_shape(action)
    elif action == 'pick_and_place':
        demo_conveyor()
    elif action == 'path':
        waypoints = action_dict.get('waypoints', [])
        if waypoints:
            waypoints.insert(0, [curr_pos[0], curr_pos[1], curr_pos[2], 0])
            execute_path(waypoints, "执行自定义轨迹", hide_approach=True)
    else:
        execute_path([curr_pos, [0, 0, -300]], "返回原点")


def async_execute_command(text):
    global is_processing
    update_status(f"状态: 解析指令 '{text}' 中...", "orange")
    curr_pos = [traj_x[-1], traj_y[-1], traj_z[-1]]
    cmd = agent.parse(text, curr_pos)

    if cmd and "error" not in cmd:
        update_status(f"状态: 解析成功", "green")
        execute_action(cmd)
    else:
        update_status(f"状态: 指令无法识别", "red")
        is_processing = False


def update_status(text, color):
    status_text_3d.set_text(text)
    status_text_3d.set_color(color)


# ================== GUI 与 交互层 ==================
root = tk.Tk()
root.title("Delta 机器人仿真平台 (高精度稳定版)")
root.geometry("1400x850")

top_frame = tk.Frame(root, padx=10, pady=5)
top_frame.pack(side=tk.TOP, fill=tk.X)

basic_control_frame = tk.LabelFrame(top_frame, text="控制模块", font=('Arial', 10, 'bold'), padx=10, pady=5)
basic_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)


def toggle_start(): global is_paused; is_paused = False; update_status("状态: 运行中", "green")


def toggle_stop(): global is_paused; is_paused = True; update_status("状态: 已暂停", "red")


def trigger_reset():
    global is_processing, current_mode
    if not is_processing:
        is_processing = True
        current_mode = 0
        execute_action({'action': 'home'})


tk.Button(basic_control_frame, text="▶ 启动", command=toggle_start, width=8, bg="#4CAF50", fg="white").pack(
    side=tk.LEFT, padx=3)
tk.Button(basic_control_frame, text="⏸ 暂停", command=toggle_stop, width=8, bg="#FF9800", fg="white").pack(side=tk.LEFT,
                                                                                                           padx=3)
tk.Button(basic_control_frame, text="↺ 复位", command=trigger_reset, width=8, bg="#F44336", fg="white").pack(
    side=tk.LEFT, padx=3)

mode_frame = tk.LabelFrame(top_frame, text="模式选择", font=('Arial', 10, 'bold'), padx=10, pady=5)
mode_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
notebook = ttk.Notebook(mode_frame)
notebook.pack(fill=tk.BOTH, expand=True)

tab_mode1 = ttk.Frame(notebook)
notebook.add(tab_mode1, text="模式1: 点对点运动")
tk.Label(tab_mode1, text="X:").pack(side=tk.LEFT, padx=2)
entry_x = tk.Entry(tab_mode1, width=6);
entry_x.insert(0, "100");
entry_x.pack(side=tk.LEFT)
tk.Label(tab_mode1, text="Y:").pack(side=tk.LEFT, padx=2)
entry_y = tk.Entry(tab_mode1, width=6);
entry_y.insert(0, "50");
entry_y.pack(side=tk.LEFT)
tk.Label(tab_mode1, text="Z:").pack(side=tk.LEFT, padx=2)
entry_z = tk.Entry(tab_mode1, width=6);
entry_z.insert(0, "-400");
entry_z.pack(side=tk.LEFT)


def execute_mode1():
    global current_mode
    try:
        x, y, z = float(entry_x.get()), float(entry_y.get()), float(entry_z.get())
        current_mode = 1
        execute_action({'action': 'path', 'waypoints': [[x, y, z, 1]]})
    except ValueError:
        pass


tk.Button(tab_mode1, text="移动", command=execute_mode1, bg="#2196F3", fg="white", width=8).pack(side=tk.LEFT, padx=10)

tab_mode2 = ttk.Frame(notebook)
notebook.add(tab_mode2, text="模式2: 画出图形")
tk.Button(tab_mode2, text="画圆", command=lambda: draw_shape('circle'), width=12).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(tab_mode2, text="画三角", command=lambda: draw_shape('triangle'), width=12).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(tab_mode2, text="画五角星", command=lambda: draw_shape('star'), width=12).pack(side=tk.LEFT, padx=5, pady=5)

tab_mode3 = ttk.Frame(notebook)
notebook.add(tab_mode3, text="模式3：自然语言处理（调用大模型）")
tk.Label(tab_mode3, text="指令:").pack(side=tk.LEFT, padx=5)
cmd_entry = tk.Entry(tab_mode3, width=45)
cmd_entry.pack(side=tk.LEFT, padx=5)
cmd_entry.insert(0, "画一只猪")


def on_nlp_submit(event=None):
    global is_processing, current_mode
    text = cmd_entry.get()
    if is_processing or not text.strip(): return
    current_mode = 3
    is_processing = True
    threading.Thread(target=async_execute_command, args=(text,), daemon=True).start()
    cmd_entry.delete(0, tk.END)


cmd_entry.bind("<Return>", on_nlp_submit)
tk.Button(tab_mode3, text="执行", command=on_nlp_submit, bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=5)

tab_mode4 = ttk.Frame(notebook)
notebook.add(tab_mode4, text="模式4：工业场景物块运输")
tk.Label(tab_mode4, text="动态生成随机料盒位置并规划抓取路径").pack(side=tk.LEFT, padx=10)
tk.Button(tab_mode4, text="启动抓取", command=demo_conveyor, bg="#00BCD4", fg="white", width=18).pack(side=tk.LEFT,
                                                                                                      padx=10)

bottom_frame = tk.Frame(root, padx=10, pady=5)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# PID 控制面板
ext1_frame = tk.LabelFrame(bottom_frame, text="扩展功能1: PID 实时调节", font=('Arial', 9, 'bold'))
ext1_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y, expand=True)


def update_pid(*args):
    pid_sys.kp = scale_kp.get()
    pid_sys.ki = scale_ki.get()
    pid_sys.kd = scale_kd.get()


scale_kp = tk.Scale(ext1_frame, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, label="Kp",
                    command=update_pid, length=120)
scale_kp.set(pid_sys.kp)  # 更新了默认的 Kp
scale_kp.pack(side=tk.LEFT, padx=5)

scale_ki = tk.Scale(ext1_frame, from_=0.0, to=0.05, resolution=0.001, orient=tk.HORIZONTAL, label="Ki",
                    command=update_pid, length=120)
scale_ki.set(pid_sys.ki)
scale_ki.pack(side=tk.LEFT, padx=5)

scale_kd = tk.Scale(ext1_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label="Kd",
                    command=update_pid, length=120)
scale_kd.set(pid_sys.kd)  # 更新了默认的 Kd
scale_kd.pack(side=tk.LEFT, padx=5)

# 工作空间可视化
ext2_frame = tk.LabelFrame(bottom_frame, text="扩展功能2: 工作空间可视化", font=('Arial', 9, 'bold'))
ext2_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y, expand=True)


def finish_ws_calculation(ws_x, ws_y, ws_z, ws_c):
    try:
        global workspace_scatter
        if ws_x:
            if 'workspace_scatter' in globals() and workspace_scatter is not None:
                try:
                    workspace_scatter.remove()
                except:
                    pass
            workspace_scatter = ax_3d.scatter(ws_x, ws_y, ws_z, c=ws_c, cmap='viridis', alpha=0.4)
            workspace_scatter.set_visible(True)
            fig.canvas.draw_idle()
        update_status("状态: 工作空间渲染完成", "green")
        btn_ws.config(text="隐藏工作空间", state=tk.NORMAL)
    except tk.TclError:
        pass


def toggle_workspace():
    global workspace_scatter
    if 'workspace_scatter' in globals() and workspace_scatter is not None and workspace_scatter.get_visible():
        workspace_scatter.set_visible(False)
        btn_ws.config(text="显示工作空间")
        fig.canvas.draw_idle()
        return

    btn_ws.config(state=tk.DISABLED, text="正在后台计算...")
    update_status("状态: 正在后台多线程计算工作空间...", "orange")

    def calc_thread():
        ws_x, ws_y, ws_z, ws_c = [], [], [], []
        for z in range(-550, -200, 25):
            for x in range(-250, 250, 25):
                for y in range(-250, 250, 25):
                    if x ** 2 + y ** 2 > 250 ** 2: continue
                    thetas = robot.inverse_kinematics(x, y, z)
                    if thetas:
                        ws_x.append(x)
                        ws_y.append(y)
                        ws_z.append(z)
                        ws_c.append(np.sum(np.array(thetas) ** 2))
        root.after(0, lambda: finish_ws_calculation(ws_x, ws_y, ws_z, ws_c))

    threading.Thread(target=calc_thread, daemon=True).start()


btn_ws = tk.Button(ext2_frame, text="计算并显示工作空间（点的深浅表示奇异性倾向）", command=toggle_workspace)
btn_ws.pack(side=tk.LEFT, padx=20, pady=10)

# 自动避障控制
ext3_frame = tk.LabelFrame(bottom_frame, text="扩展功能3: 自动避障", font=('Arial', 9, 'bold'))
ext3_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y, expand=True)


def toggle_obs():
    global show_obstacle
    show_obstacle = not show_obstacle
    obs_mesh.set_visible(show_obstacle)
    btn_obs.config(text=f"避障圆球: {'开启' if show_obstacle else '关闭'}")


btn_obs = tk.Button(ext3_frame, text="避障圆球: 开启", command=toggle_obs, width=18)
btn_obs.pack(side=tk.LEFT, padx=20, pady=10)

# ================== 绘图与动画层 ==================
fig = plt.Figure(figsize=(14, 7))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.3, 1])

ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_p = fig.add_subplot(gs[0, 1])
ax_v = fig.add_subplot(gs[1, 1])
ax_a = fig.add_subplot(gs[2, 1])
fig.tight_layout(pad=3.0)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

line_tp_z, = ax_p.plot([], [], 'r--', label='目标角度')
line_ap_z, = ax_p.plot([], [], 'b-', label='实际角度(跟随)')
line_tv, = ax_v.plot([], [], 'k--', label='目标角速度')
line_av, = ax_v.plot([], [], 'b-', label='实际角速度')
line_ta, = ax_a.plot([], [], 'r--', label='目标角加速度')
line_aa, = ax_a.plot([], [], 'g-', label='实际角加速度')

ax_p.legend(loc='upper right')
ax_p.set_title("关节1 角度跟踪对比 (deg)")
ax_v.legend(loc='upper right')
ax_v.set_title("关节1 角速度 (deg/frame)")
ax_a.legend(loc='upper right')
ax_a.set_title("关节1 角加速度 (deg/frame^2)")

base_line, = ax_3d.plot([], [], [], 'k-', linewidth=3)
end_line, = ax_3d.plot([], [], [], 'g-', linewidth=3)
history_line, = ax_3d.plot([], [], [], 'orange', linestyle='-', linewidth=2.5)
target_point, = ax_3d.plot([], [], [], 'ro', markersize=6, alpha=0.5)

arms_main = [ax_3d.plot([], [], [], '#d62728', linewidth=6)[0] for _ in range(3)]
arms_sub_L = [ax_3d.plot([], [], [], '#1f77b4', linewidth=2)[0] for _ in range(3)]
arms_sub_R = [ax_3d.plot([], [], [], '#1f77b4', linewidth=2)[0] for _ in range(3)]

belt_line, = ax_3d.plot([], [], [], color='#00BCD4', linewidth=3, visible=False)
box_line, = ax_3d.plot([], [], [], color='#8D6E63', linewidth=3, visible=False)
workpiece, = ax_3d.plot([], [], [], marker='s', color='gold', markersize=14, markeredgecolor='black', visible=False)
workspace_scatter = ax_3d.scatter([], [], [], c=[], cmap='viridis', alpha=0.15, visible=False)

u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
obs_mesh = ax_3d.plot_wireframe(obs_center[0] + obs_r * np.cos(u) * np.sin(v),
                                obs_center[1] + obs_r * np.sin(u) * np.sin(v),
                                obs_center[2] + obs_r * np.cos(v), color='r', alpha=0.4)

ax_3d.set_xlim([-300, 300])
ax_3d.set_ylim([-300, 300])
ax_3d.set_zlim([-600, 0])
ax_3d.set_title("Delta 机器人", fontsize=14, fontweight='bold')
status_text_3d = ax_3d.text2D(0.05, 0.98, "状态: 就绪\n动作: 无", transform=ax_3d.transAxes, color='green', fontsize=12,
                              fontweight='bold', verticalalignment='top')


def safe_ylim(ax, d1, d2, margin=2.0):
    arr = np.concatenate([np.atleast_1d(d1), np.atleast_1d(d2)])
    arr = np.array(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0: return
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-3: mx += 1.0; mn -= 1.0
    ax.set_ylim(mn - margin, mx + margin)


def update(frame):
    global current_frame, steady_counter
    if is_paused: return base_line,

    if not is_processing and current_frame < total_frames - 1:
        current_frame += 1

    idx = min(current_frame, total_frames - 1)
    t_x, t_y, t_z = traj_x[idx], traj_y[idx], traj_z[idx]

    act_pos, act_v, act_a = pid_sys.update([t_x, t_y, t_z])

    is_idle = (current_frame >= total_frames - 1) and (np.linalg.norm(pid_sys.vel) < 0.5)

    if is_idle:
        steady_counter += 1
    else:
        steady_counter = 0

    if current_frame == clear_history_frame:
        draw_h["x"].clear()
        draw_h["y"].clear()
        draw_h["z"].clear()

    if steady_counter < 5:
        ths = robot.inverse_kinematics(act_pos[0], act_pos[1], act_pos[2])
        th1 = np.degrees(ths[0]) if ths else (actual_th_h[-1] if actual_th_h else 0)

        dth1 = th1 - actual_th_h[-1] if actual_th_h else 0
        ddth1 = dth1 - actual_dth_h[-1] if actual_dth_h else 0

        actual_th_h.append(th1)
        actual_dth_h.append(dth1)
        actual_ddth_h.append(ddth1)
        target_th_h.append(target_th_profile[idx] if len(target_th_profile) > 0 else 0)

        actual_pos_h["x"].append(act_pos[0])
        actual_pos_h["y"].append(act_pos[1])
        actual_pos_h["z"].append(act_pos[2])

        if stop_draw_frame == -1 or current_frame < stop_draw_frame:
            if traj_p[idx] == 1:
                draw_h["x"].append(act_pos[0])
                draw_h["y"].append(act_pos[1])
                draw_h["z"].append(act_pos[2])
            else:
                if len(draw_h["x"]) > 0 and not np.isnan(draw_h["x"][-1]):
                    draw_h["x"].append(np.nan)
                    draw_h["y"].append(np.nan)
                    draw_h["z"].append(np.nan)

    if show_env_mode4 and traj_g[idx]:
        workpiece.set_data([act_pos[0]], [act_pos[1]])
        workpiece.set_3d_properties([act_pos[2] - 25])

    plot_len = min(200, len(actual_th_h))
    if plot_len > 0:
        disp_tp = target_th_h[-plot_len:]
        disp_ap = actual_th_h[-plot_len:]
        disp_av = actual_dth_h[-plot_len:]
        disp_aa = actual_ddth_h[-plot_len:]
        time_axis = list(range(len(actual_th_h) - plot_len, len(actual_th_h)))

        line_ap_z.set_data(time_axis, disp_ap)

        if len(target_th_profile) > 0:
            line_tp_z.set_data(range(len(target_th_profile)), target_th_profile)
            line_tv.set_data(range(len(target_dth_profile)), target_dth_profile)
            line_ta.set_data(range(len(target_ddth_profile)), target_ddth_profile)

        line_av.set_data(time_axis, disp_av)
        line_aa.set_data(time_axis, disp_aa)

        current_len = len(actual_th_h)
        x_min = max(0, current_len - 200)
        x_max = max(50, current_len, total_frames)
        if x_max - x_min < 50: x_max = x_min + 50

        ax_p.set_xlim(x_min, x_max)
        ax_v.set_xlim(x_min, x_max)
        ax_a.set_xlim(x_min, x_max)

        safe_ylim(ax_p, target_th_profile if len(target_th_profile) > 0 else [], disp_ap)
        safe_ylim(ax_v, target_dth_profile if len(target_dth_profile) > 0 else [], disp_av)
        safe_ylim(ax_a, target_ddth_profile if len(target_ddth_profile) > 0 else [], disp_aa)

    thetas = robot.inverse_kinematics(act_pos[0], act_pos[1], act_pos[2])
    if not thetas: return base_line,

    base_pts, elbow_pts, end_pts = [], [], []
    for i in range(3):
        a, t = robot.angles[i], thetas[i]
        bx, by, bz = robot.R * np.cos(a), robot.R * np.sin(a), 0
        ex, ey, ez = bx + robot.L1 * np.cos(t) * np.cos(a), by + robot.L1 * np.cos(t) * np.sin(
            a), bz - robot.L1 * np.sin(t)
        px, py, pz = act_pos[0] + robot.r * np.cos(a), act_pos[1] + robot.r * np.sin(a), act_pos[2]
        base_pts.append([bx, by, bz])
        elbow_pts.append([ex, ey, ez])
        end_pts.append([px, py, pz])

    base_pts, elbow_pts, end_pts = np.array(base_pts), np.array(elbow_pts), np.array(end_pts)
    base_c = np.vstack((base_pts, base_pts[0]))
    base_line.set_data(base_c[:, 0], base_c[:, 1])
    base_line.set_3d_properties(base_c[:, 2])
    end_c = np.vstack((end_pts, end_pts[0]))
    end_line.set_data(end_c[:, 0], end_c[:, 1])
    end_line.set_3d_properties(end_c[:, 2])

    history_line.set_data(draw_h["x"], draw_h["y"])
    history_line.set_3d_properties(draw_h["z"])
    target_point.set_data([t_x], [t_y])
    target_point.set_3d_properties([t_z])

    for i in range(3):
        arms_main[i].set_data([base_pts[i, 0], elbow_pts[i, 0]], [base_pts[i, 1], elbow_pts[i, 1]])
        arms_main[i].set_3d_properties([base_pts[i, 2], elbow_pts[i, 2]])
        offset = (robot.W / 2) * np.array([-np.sin(robot.angles[i]), np.cos(robot.angles[i]), 0])
        e_L, e_R = elbow_pts[i] + offset, elbow_pts[i] - offset
        p_L, p_R = end_pts[i] + offset, end_pts[i] - offset
        arms_sub_L[i].set_data([e_L[0], p_L[0]], [e_L[1], p_L[1]])
        arms_sub_L[i].set_3d_properties([e_L[2], p_L[2]])
        arms_sub_R[i].set_data([e_R[0], p_R[0]], [e_R[1], p_R[1]])
        arms_sub_R[i].set_3d_properties([e_R[2], p_R[2]])

    return base_line, end_line, history_line, target_point, *arms_main, *arms_sub_L, *arms_sub_R, line_tp_z, line_ap_z, line_tv, line_av, line_ta, line_aa


ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
root.mainloop()