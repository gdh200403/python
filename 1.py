print("hello world!")


def pascal_triangle(i, j):
    """
    返回杨辉三角第i行第j列的数
    """
    # 创建一个二维数组存储杨辉三角的值
    triangle = [[1] * (n + 1) for n in range(i)]

    # 递推计算杨辉三角
    for row in range(2, i):
        for col in range(1, row):
            triangle[row][col] = triangle[row -
                                          1][col - 1] + triangle[row - 1][col]

    # 返回第i行第j列的数
    return triangle[i - 1][j - 1]


print(pascal_triangle(4, 3))
