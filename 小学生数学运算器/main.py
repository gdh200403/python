import random


def generate_question(mode):
    """根据模式生成随机的题目"""
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    if mode == "+":
        operator = "+"
        answer = num1 + num2
    elif mode == "-":
        operator = "-"
        while num1 < num2:   # 确保结果为正数
            num1 = random.randint(1, 100)
            num2 = random.randint(1, 100)
        answer = num1 - num2
    elif mode == "*":
        operator = "*"
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 20)
        answer = num1 * num2
    elif mode == "/":
        operator = "/"
        while num1 % num2 != 0 or num1/num2 ==1 or num2 ==1:  # 确保结果为整数
            num1 = random.randint(1, 100)
            num2 = random.randint(1, 100)
        answer = num1 // num2
    else:  # mode == "challenge"
        operator = random.choice(["+", "-", "*", "/"])
        if operator == "+":
            answer = num1 + num2
        elif operator == "-":
            while num1 < num2:   # 确保结果为正数
                num1 = random.randint(1, 100)
                num2 = random.randint(1, 100)
            answer = num1 - num2
        elif operator == "*":
            num1 = random.randint(1, 10)
            num2 = random.randint(1, 20)
            answer = num1 * num2
        else:
            while num1 % num2 != 0 or num1/num2 ==1 or num2 ==1:  # 确保结果为整数
                num1 = random.randint(1, 100)
                num2 = random.randint(1, 100)
            answer = num1 // num2
    return f"{num1} {operator} {num2} =", answer, mode


def check_answer(answer, uanswer):
    """检查答案是否正确"""
    return answer == uanswer


import time
import sys

def print_sleep (content, sleepseconds=4):
    print(content)
    time.sleep(sleepseconds)

def prefix():
    print_sleep('\n郭xx小朋友，你好！')
    print_sleep('欢迎你来到数学的世界！')
    print_sleep('我是数学王国的一个小小计算器')
    print_sleep('加法、减法、乘法、除法都是我的好朋友！')
    print_sleep('今天我带着他们一起，跨过了千山万水到来到了这台电脑旅游！\n')
    print_sleep('你们人类真是聪明，竟然发明了电脑这个神奇的东西~')
    print_sleep('我想坐在这台电脑前的你，一定也是个聪明的小朋友吧！\n')
    print_sleep('不过，我要告诉你',2)
    print_sleep('没有我们数学王国的朋友们',2)
    print_sleep('你们人类再聪明也造不出电脑呢')
    print_sleep('好啦，有关我们数学王国和电脑的秘密，以后再和你说\n')
    print_sleep('今天,我特地来找你玩儿！',2)
    print_sleep('因为，我想和你交朋友！')
    print('如果你愿意和我交朋友，请和我打个招呼吧！')
    sys.stdout.flush()
    input()
    print_sleep('嗨，很高兴和你交朋友！')
    print_sleep('瞧，我的加减乘除那几个小伙伴也来了，他们也想和你交朋友呢！')
    print('如果你愿意和他们交朋友，请和他们也打个招呼吧！')
    sys.stdout.flush()
    input()
    print_sleep('嗨，他们说很高兴和你交朋友！')
    print_sleep('那么，聪明的你，请勇敢接受我们的挑战吧！')
    print('在这个挑战中，你会和我的四个好伙伴做游戏')
    print('你要记住，如果你玩累了，在任何算式后输入一句咒语')
    print('q',end='')
    time.sleep(1)
    print('u',end='')
    time.sleep(1)
    print('i',end='')
    time.sleep(1)
    print('t')
    time.sleep(1)
    print_sleep('这是一个英文单词，意思是退出，输入它就可以结束我们的游戏！我和我的四个小伙伴会一起算出你的游戏表现！')
    print_sleep('好啦！游戏开始！')

a = input('嗨，郭xx小朋友，这是我们第一次见面吗？\n是请按1，不是请按0:')
if(int(a)):
    prefix()
else:
    print('游戏开始！请继续你的挑战！')
while True:
    print('---------------------------------------------------')
    print("(注意：乘号用“*”表示，除号用“/”表示。)")
    mode = input("请选择模式( + | - | * | / |挑战模式 ):")
    if mode not in ["+", "-", "*", "/", "挑战模式"]:
        print("要在键盘上找到我的小伙伴对应的符号噢！不然那四个家伙会生气的！再来一次！")
        continue
    # 开始计时
    start_time = time.perf_counter()
    # 初始化统计信息
    total_questions = 0
    correct_questions = 0
    score = 0
    # 出题循环
    while True:
        # 生成题目
        question, answer, mode = generate_question(mode)

        
        # 提示用户输入答案
        user_answer = input(question)
        
        # 如果用户输入quit，则退出程序
        if user_answer.lower() == "quit":
            break
        
        # 如果用户输入不是数字，则重新出题
        elif not user_answer.isnumeric():
            print("要按我们数学王国的规矩输入，请输入数字！")
            continue
        
        # 校验答案，并统计做对的题数
        elif check_answer(answer,int(user_answer)):
            correct_questions += 1
            score+=1
            print("回答正确！你真棒！")
        else:
            print("回答错误！再加把劲！")
            print('正确答案应该是:',answer)
            score-=1
                
        # 统计总题数
        total_questions += 1

        if mode != "挑战模式":  # 如果不是挑战模式，则显示得分
            print(f"当前得分：{score}\n")
            
    # 结束计时
    end_time = time.perf_counter()
    # 统计答题情况
    accuracy = correct_questions / total_questions if total_questions > 0 else 0
    # 打印统计信息
    print(f"\n郭xx小朋友，本次你共做了{total_questions}道题")
    print(f"做对了{correct_questions}道题")
    print(f"你的得分是：{score}")
    print(f"正确率为{accuracy:.0%}")
    print(f"共花费时间{end_time-start_time:.2f}秒")
    print(f"平均每道题目你只花了{(end_time-start_time)/total_questions:.2f}秒完成！")
    # 根据正确率给出评价
    if accuracy == 1:
        print("太棒了，你真是个小算盘！")
    elif accuracy >= 0.9:
        print("非常好，你已经是一个小算盘了！")
    elif accuracy >= 0.7:
        print("不错，你已经掌握了很多技巧！")
    elif accuracy >= 0.5:
        print("还需努力，多加练习，相信你会更厉害！")
    else:
        print("加油，你一定能行！")
    print('')
    option=input('还想再来一次吗？输入yes继续，输入no退出:')
    if option=='yes':
        pass
    else:
        quit()