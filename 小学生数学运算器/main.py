import random
import time
import sys
import colorama


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


def printer(text,delay=0.2,sleepseconds=2):
    print()
    for char in text:
        print(char,end='',flush=True)#使用flush=True强制立即输出，避免控制台缓冲区的影响
        time.sleep(delay)
    time.sleep(sleepseconds)


def random_color_printer(text,delay=0.5,sleepseconds=2):
    #定义颜色列表
    colors=[colorama.Fore.RED,colorama.Fore.GREEN,colorama.Fore.YELLOW,colorama.Fore.BLUE,colorama.Fore.MAGENTA,colorama.Fore.CYAN]
    print()
    for char in text:
        color=random.choice(colors)
        print(f"{color}{char}"+colorama.Fore.RESET,end='',flush=True)
        time.sleep(delay)
    time.sleep(sleepseconds)


def print_sleep (content, sleepseconds=4):
    print(content)
    time.sleep(sleepseconds)


def prefix():
    random_color_printer('郭钰涵小朋友，你好！')
    random_color_printer('欢迎你来到数学的世界！')
    printer('\n我是数学王国的一个小算盘')
    printer(colorama.Fore.GREEN+'加法'+colorama.Fore.RESET+'、'+colorama.Fore.RED+'减法'+colorama.Fore.RESET+'、'+colorama.Fore.BLUE+'乘法'+colorama.Fore.RESET+'、'+colorama.Fore.YELLOW+'除法'+colorama.Fore.RESET+'都是我的好朋友！')
    printer('今天我带着他们一起，跨过了千山万水到来到了这台电脑旅行！\n')
    printer('电脑真神奇，你们人类能发明它，真是聪明！')
    printer('我想坐在这台电脑前的你，一定也是个聪明的小朋友吧！\n')
    printer('不过，我要告诉你:')
    printer('没有我们数学王国的朋友们')
    printer('你们人类再聪明，也造不出电脑呢')
    printer('好啦，有关我们数学王国和电脑的秘密，以后再和你说\n')
    printer('今天,我特地来找你玩儿！')
    printer('因为，我想和你'+colorama.Fore.GREEN+'交朋友！'+colorama.Fore.RESET)
    printer('如果你愿意和我交朋友，请和我打个招呼吧！\n',sleepseconds=0)
    sys.stdout.flush()
    input()
    printer('嗨，很高兴认识你！')
    printer('瞧，我的加减乘除那几个小伙伴也来了，他们也想和你交朋友呢！')
    printer('如果你愿意和他们交朋友，请和他们也打个招呼吧！\n',sleepseconds=0)
    sys.stdout.flush()
    input()
    printer('嗨，他们说很高兴和你交朋友！')
    printer('那么，聪明的你，请勇敢接受我们的挑战吧！')
    printer('在这个挑战中，你会和我的四个好伙伴做游戏')
    printer('你要记住，如果你玩累了，在任何算式后输入一句咒语')
    printer(colorama.Fore.YELLOW+'quit'+colorama.Fore.RESET,delay=1)
    printer('这是一个英文单词，意思是退出，输入它就可以'+colorama.Fore.RED+'结束'+colorama.Fore.RESET+'我们的游戏！我和我的四个小伙伴会一起算出你的游戏表现！')
    printer('好啦！游戏开始！')


colorama.init()
print(colorama.Style.BRIGHT,colorama.Back.BLACK)
a = input('嗨，郭钰涵小朋友，这是我们第一次见面吗？\n是请按1，不是请按0:\n')
if(int(a)):
    prefix()
else:
    random_color_printer('游戏开始！请继续你的挑战！',delay=0.2)
while True:
    printer('---------------------------------------------------',delay=0.02,sleepseconds=1)
    print("\n(注意："+colorama.Fore.BLUE+"乘号"+colorama.Fore.RESET+"用"+colorama.Fore.BLUE+'*'+colorama.Fore.RESET+"表示，"+colorama.Fore.YELLOW+"除号"+colorama.Fore.RESET+"用"+colorama.Fore.YELLOW+'/'+colorama.Fore.RESET+"表示。)")
    mode = input("请选择模式("+colorama.Fore.GREEN+'+'+colorama.Fore.RESET+'|'+colorama.Fore.RED+'-'+colorama.Fore.RESET+'|'+colorama.Fore.BLUE+'*'+colorama.Fore.RESET+'|'+colorama.Fore.YELLOW+'/'+colorama.Fore.RESET+'|'+colorama.Fore.MAGENTA+"挑战模式"+colorama.Fore.RESET+" ):")
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
            print(colorama.Fore.GREEN+"回答正确！"+colorama.Fore.RESET+"你真棒！")
        else:
            print(colorama.Fore.RED+"回答错误！"+colorama.Fore.RESET+"再加把劲！")
            print(colorama.Fore.GREEN+'正确答案'+colorama.Fore.RESET+'应该是:'+ colorama.Fore.YELLOW+str(answer)+colorama.Fore.RESET)
            score-=1
                
        # 统计总题数
        total_questions += 1

        if mode != "挑战模式":  # 如果不是挑战模式，则显示得分
            print(f"当前得分：{score}\n") if score>=0 else print(colorama.Fore.RED+f"当前得分：{score}\n"+colorama.Fore.RESET)
            
    # 结束计时
    end_time = time.perf_counter()
    # 统计答题情况
    accuracy = correct_questions / total_questions if total_questions > 0 else 0
    # 打印统计信息
    printer(f"郭钰涵小朋友，本次你共做了{total_questions}道题",delay=0.05,sleepseconds=1)
    printer(f"做对了{correct_questions}道题",delay=0.05,sleepseconds=1)
    printer(f"你的得分是：{score}",delay=0.05,sleepseconds=1)
    printer(f"正确率为{accuracy:.0%}",delay=0.05,sleepseconds=1)
    printer(f"共花费时间{end_time-start_time:.2f}秒",delay=0.05,sleepseconds=1)
    printer(f"平均每道题目你只花了{(end_time-start_time)/total_questions:.2f}秒完成！\n",delay=0.05,sleepseconds=1)
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