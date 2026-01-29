def calculate_dof(num_links, num_joints, joint_dofs):
    m = 6# 3D 공간에서 링크가 가질 수 있는 최대 자유도는 6
    # Kutzbach Formula 적용
    F = m * (num_links - num_joints - 1) + sum(joint_dofs)
    return F

# --- 메인 실행 ---
print("자유도 계산기")

try:
    links = int(input("링크의 개수를 입력하세요: "))
    joints = int(input("조인트의 개수를 입력하세요: "))
    
    dofs = []
    for i in range(1, joints+1):
        dofs.append(int(input(f"{i}번 관절의 자유도를 입력하세요: ")))

    result_dof = calculate_dof(links, joints, dofs)
    print(f"계산된 자유도: {result_dof}")

except ValueError:
    print("오류: 숫자를 입력해주세요.")