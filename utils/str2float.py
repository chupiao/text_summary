from functools import reduce

def str2float(s):
    '''
    将带底数e的字符串转换为float类型
    :param s: 字符串
    :return: float类型的数
    '''
    s = s.split('.')
    a = s[0]
    b = s[1]
    if a[0] == '-':
        a = a[1:]
        front = reduce(lambda x,y:y+x*10,map(int,a))
        a = 0
        if 'e' in b:
            for i in b:
                a += 1
                if i == 'e':
                    c = b[a+1:]
                    middle = reduce(lambda x,y:y+x*10,map(int,c))
                    b = b[:a-1]
                    buttom = reduce(lambda x,y:y+x*10,map(int,b))
                    result = (front + buttom / 10 ** (len(b))) / 10 ** middle
                    result = -result
                    return result
        else:
            buttom = reduce(lambda x, y: y + x * 10, map(int, b))
            result = front + buttom / 10 ** (len(b))
            result = -result
            return result
    else :
        front = reduce(lambda x, y: y + x * 10, map(int, a))
        a = 0
        if 'e' in b:
            for i in b:
                a += 1
                if i == 'e':
                    c = b[a+1:]
                    middle = reduce(lambda x,y:y+x*10,map(int,c))
                    b = b[:a-1]
                    buttom = reduce(lambda x,y:y+x*10,map(int,b))
                    result = (front + buttom / 10 ** (len(b))) / 10 ** middle
                    return result
        else:
            buttom = reduce(lambda x, y: y + x * 10, map(int, b))
            result = front + buttom / 10 ** (len(b))
            return result