import numpy as np

# 定义函数设置无人机坐标
def drone_initialize():
    name=input("请设置无人机代号：")
    x=float(input("请设置x坐标:"))
    y=float(input("请设置y坐标:"))
    z=float(input("请设置z坐标:"))
    speed=float(input(f"请设置无人机{name}速度:"))
    r=float(input(f"请设置无人机{name}扫描半径:"))
    print(f"无人机'{name}'坐标已设置为:({x},{y},{z}),速度已设置为{speed}，扫描半径已设置为{r}")
    return np.array([x,y,z]),speed,r


# dro_pos1=drone_initialize()
# dro_pos2=drone_initialize()
# dro_pos3=drone_initialize()



# pos=np.array([dro_pos1,dro_pos2,dro_pos3])
# ava_x=round(np.mean(pos[:,0]),2)
# ava_y=round(np.mean(pos[:,1]),2)
# ava_z=round(np.mean(pos[:,2]),2)
# print(f"三架无人机平均坐标为{ava_x},{ava_y},{ava_z}")


def calculate_distance(np1,np2):
    difference=np1-np2
    distance=np.sqrt(np.sum(difference**2))
    return round(distance,2)


# dis_1_2=calculate_distance(dro_pos1,dro_pos2)
# dis_2_3=calculate_distance(dro_pos2,dro_pos3)
# dis_1_3=calculate_distance(dro_pos3,dro_pos1)
# print(dis_1_2)
# print(dis_2_3)
# print(dis_1_3)
