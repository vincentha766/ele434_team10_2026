# 弧形避障 (Bug-0 Sidestep) 完整讲解

## 一、为什么要"弧形"？

差速小车有两种基本避障方式：

```
方式 A: 停下 → 原地转 → 直走          方式 B: 边走边转 (弧形)

  ┌──┐                                   ┌──┐
  │障│   停→转→走→转→走                   │障│     ╭───╮
  │碍│   效率低, 路径锯齿                 │碍│    ╱     ╲   平滑弧线
  │物│                                   │物│   ╱       ╲  不停车
  └──┘                                   └──┘  ╱         ╲
```

方式 B 就是弧形避障 — **同时给线速度 v 和角速度 ω**，小车走出一条弧线绕过障碍物，更快更平滑。

---

## 二、弧形运动的物理原理

差速小车的运动学：
```
弧形半径 R = v / ω

代码中的参数:
  SIDESTEP_V = 0.06 m/s  (线速度, 很慢)
  SIDESTEP_W = 1.20 rad/s (角速度, 较快)

  R = 0.06 / 1.20 = 0.05m (半径 5cm)
```

- 半径越小 → 转弯越急 → 绕障越紧凑但可能擦到障碍物
- 半径越大 → 转弯越缓 → 更安全但绕路更远

---

## 三、状态机逐层拆解

整个避障逻辑分为 4 个阶段：

### 阶段 1: 触发条件 — "前方危险！"

`work.py:404`:
```python
front_danger = min_f < SAFE_RADIUS   # SAFE_RADIUS = 0.32m
```

当正前方 ±22° 范围内最近的障碍物距离 < 0.32m 时，触发避障。

**为什么是 0.32m？** TurtleBot3 Waffle 外接圆半径约 0.21m + 0.11m 安全裕量 = 0.32m。这个距离能保证从检测到刹住车，不会撞上。

### 阶段 2: 选方向 — "往哪边绕？"

`work.py:423-439`，决策逻辑如下：

```
                    目标点在左边？(angle_error > 0)
                   /                    \
                 是                      否
                /                        \
         左边够空吗？                 右边够空吗？
    (left_clear > right-0.30)    (right_clear > left-0.30)
          |                            |
         是 → 左转(+1)              是 → 右转(-1)
         否 → 哪边空去哪边          否 → 哪边空去哪边
```

**核心原则**: 优先朝目标方向转（绕完更接近目标），但如果那一侧太挤（差距超过 0.30m），就往空旷的一侧转。

```python
toward_left = angle_error > 0          # 目标在左边吗？
left_clear  = min_fl + min_l           # 左侧总空旷度
right_clear = min_fr + min_r           # 右侧总空旷度

# 0.30m 的容忍度: 即使目标侧稍微窄一点, 也优先往目标侧转
if toward_left and left_clear > right_clear - 0.30:
    avoid_dir = 1    # 左转, 障碍保持在右侧
```

**`avoid_dir` 的含义**: `+1` = 左转（逆时针），`-1` = 右转（顺时针）。一旦选定就**承诺不变**，防止在两侧反复摇摆。

### 阶段 3: 执行绕行 — "怎么走弧线？"

`work.py:442-449`，有两种子模式：

```python
if avoid_active:
    if min_f < 0.22:           # 太近了! (< 22cm)
        linear_vel = 0.0       #   停车
        angular_vel = avoid_dir * p_max_w  #   全速原地转 (1.82 rad/s)
    else:                      # 前方有一点空间
        linear_vel = SIDESTEP_V   #   慢慢走 (0.06 m/s)  <- 弧线的"线"
        angular_vel = avoid_dir * SIDESTEP_W  #   快速转 (1.20 rad/s) <- 弧线的"弧"
```

画面感：
```
场景 A: 距离 < 22cm (紧急)       场景 B: 距离 22cm~32cm (正常弧形)

     ┌──┐                              ┌──┐
     │障│  <- 只有 20cm!               │障│  <- 还有 28cm
     │碍│                              │碍│
     └──┘                              └──┘
      车  原地全速旋转                   车  边走边转
         v=0, w=max                        v=0.06, w=1.20
```

### 阶段 4: 退出条件 — "障碍绕过了吗？"

`work.py:407-421`，有两个退出条件（满足任一即可）：

```python
# 条件 A: 已经转够角度 (>30°) 且前方清空
exit_on_rotate = (rotated > 0.5 and min_f > 0.37)

# 条件 B: 已走出一段距离 (>25cm) 且跟随侧开阔 且前方清空
exit_on_pass = (traveled > 0.25 and side_min > 0.45 and min_f > 0.37)
```

**"跟随侧"的概念**：
```
左转绕障 (avoid_dir = +1):
  障碍物在右边 → 监控右前+正右 (min_r, min_fr)
  当右侧 > 0.45m → 说明障碍物已经被甩在身后了

      ╭──→ 前进方向
     ╱
    车          ┌──┐
    监控右侧→   │障│  <- 这一侧的距离在增大 = 绕过去了
               └──┘
```

---

## 四、尾部保护

`work.py:457-462` — 原地旋转时，车尾可能撞到后方障碍物：

```python
if linear_vel < 0.05 and abs(angular_vel) > 0.2:  # 在原地转
    rear_min = min(min_bl, min_b, min_br, min_l, min_r)
    if rear_min < SAFE_RADIUS:
        gain = max(0.0, (rear_min - 0.21) / (SAFE_RADIUS - 0.21))
        angular_vel *= gain   # 后方越近, 角速度越小, 直到停转
```

这是一个线性衰减：
```
gain
1.0 |----------\
    |           \
0.5 |            \
    |             \
0.0 |--------------\---
    0.21m    0.28m  0.32m   后方距离
    (车身)   (减速) (安全)
```

---

## 五、参数总览

| 参数 | 值 | 含义 | 调大的效果 | 调小的效果 |
|------|----|------|-----------|-----------|
| `SAFE_RADIUS` | 0.32m | 触发避障的距离 | 更早避障，更保守 | 更晚避障，可能擦碰 |
| `SIDESTEP_V` | 0.06 m/s | 弧形线速度 | 弧半径大，绕路远 | 弧半径小，转弯急 |
| `SIDESTEP_W` | 1.20 rad/s | 弧形角速度 | 转得快，弧半径小 | 转得慢，弧半径大 |
| `AVOID_MIN_ROT` | 0.5 rad | 最小承诺转角 | 每次绕更多再退出 | 容易过早退出 |
| `AVOID_EXIT_SIDE` | 0.45m | 跟随侧清空阈值 | 需要更空才退出 | 更容易退出避障 |

---

## 六、动手练习

### 练习 1: 最简弧形避障（在 obstacle.py 中实现）
只有前方检测 + 固定左转弧线，不带航点导航：
```python
def juggment(self, dist):
    lon = len(dist)
    i = lon / 360

    front = dist[0:int(22*i)] + dist[int(338*i):]
    min_front = min(r if 0.05 < r else 3.5 for r in front)

    if min_front < 0.32:
        # 弧形: 慢走 + 快转
        self.vel_cmd.twist.linear.x = 0.06
        self.vel_cmd.twist.angular.z = 1.2   # 固定左转
        self.pub.publish(self.vel_cmd)
    else:
        self.move(0.2)
```

### 练习 2: 加入方向选择
比较左/右侧空旷度，选择更空的一侧转弯。

### 练习 3: 加入退出条件和"承诺方向"机制
记录进入避障时的 yaw 角和位置，监控跟随侧距离变化来判断退出。
