'''
@Author       : Donghao Guo, USTC
@Date         : 2024-1-24 20:18
@LastEditors  : Donghao Guo
@LastEditTime : 2024-1-24 20:18
@Description  : 用于统计学生参与活动情况的工具
'''
import os
import re
import pandas as pd

# 初始化字典
activity_dict = {}
student_dict = {}

# 遍历目录和子目录
for root, dirs, files in os.walk('23秋参与统计'):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            # 提取学号和姓名
            match = re.match(r'(PB\d{8})([\u4e00-\u9fa5]+)\((\d+)\)\.(jpg|png)', file)
            if match:
                student_id_name = match.group(1) + match.group(2)
                student_id = match.group(1)
                student_name = match.group(2)
                activity_name = os.path.basename(root)
                
                # 更新活动字典
                if activity_name not in activity_dict:
                    activity_dict[activity_name] = []
                activity_dict[activity_name].append(student_id_name)
                
                # 更新学生字典
                if (student_id, student_name) not in student_dict:
                    student_dict[(student_id, student_name)] = {'count': 0, 'activities': []}
                if activity_name not in student_dict[(student_id, student_name)]['activities']:
                    student_dict[(student_id, student_name)]['count'] += 1
                    student_dict[(student_id, student_name)]['activities'].append(activity_name)

# 定义一个字典，映射汉字数字到阿拉伯数字
chinese_to_arabic = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

def get_number_from_activity_name(activity_name):
    # 提取活动名称中的汉字数字
    chinese_number = activity_name[1]
    # 返回对应的阿拉伯数字
    return chinese_to_arabic[chinese_number]

# 按照活动名称中的汉字数字排序
activity_dict = dict(sorted(activity_dict.items(), key=lambda x: get_number_from_activity_name(x[0])))
# 给每个学生的参与活动列表按照活动名称中的汉字数字排序
for k, v in student_dict.items():
    v['activities'].sort(key=lambda x: get_number_from_activity_name(x))
# 给每个活动的参与同学按照学号排序
for k, v in activity_dict.items():
    v.sort()

# 创建DataFrame
df1 = pd.DataFrame([(k, '、'.join(v)) for k, v in activity_dict.items()], columns=['活动名称', '参与同学'])
df2 = pd.DataFrame([(k[0], k[1], v['count'], '、'.join(v['activities'])) for k, v in student_dict.items()], columns=['学号', '姓名', '参与次数', '参与的子活动名称'])

# 写入Excel文件
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='活动参与者', index=False)
    df2.to_excel(writer, sheet_name='学生参与情况', index=False)