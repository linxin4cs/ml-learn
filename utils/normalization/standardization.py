import math as mt


def get_avg(data):
    """
    求平均数

    :param data: 将进行求平均值的序列

    :return 平均数
    """

    sum_of_nums = 0

    for num in data:
        sum_of_nums += num

    avg = sum_of_nums / len(data)

    return avg


def get_std_dev(data):
    """
    求标准差

    :param data: 进行求标准差的序列，一维数组

    :return 标准差
    """

    avg = get_avg(data)
    length = len(data)
    sum_of_nums = 0

    for num in data:
        sum_of_nums += (avg - num) ** 2

    std_dev = mt.sqrt(sum_of_nums / length)

    return std_dev


def get_avgs_and_std_devs(dataset):
    """
    计算给定数据集的平均数和标准差

    :param dataset: 将进行计算平均数和标准差的数据集，二维数组，形如 [[1, 2, 3, ...], [3, 4, 5], ...]

    :return
    通过给定数据集计算得到的平均数和标准差，字典，形如 { "avgs": [1, 2, 3, ...], "standard_deviations": [1, 2, 3, ...] }
    """

    # 标准差序列以及平均数序列
    std_devs = []
    avgs = []

    for i in range(len(dataset[0])):
        # 用于保存计算标准差的数据序列
        items = []

        for j in range(len(dataset)):
            # 将当前维度的数据保存在 items 中
            items.append(dataset[j][i])

        std_devs.append(get_std_dev(items))
        avgs.append(get_avg(items))

    return {
        "avgs": avgs,
        "std_devs": std_devs
    }


def get_normed_data(data, avgs_and_std_devs):
    """
    对数据进行标准化

    :param data: 将进行标准化的数据，一维数组，形如 [1, 2, 3, 4]，data[0], ...
    均有对应的平均数 avgs_std_devs["avgs"][0] 和标准差 avgs_std_devs["std_devs"][0]
    :param avgs_and_std_devs: 数据中每位元素对应计算中的平均数和标准差，字典，形如 { "avgs": [1, 2, 3, ...], "std_devs": [1, 2, 3, ...] }

    :return: 标准化后的数据，一维数组，形如 [1, 2, 3, 4]
    """

    res = []

    for i in range(len(data)):
        val = (data[i] - avgs_and_std_devs["avgs"][i]) / avgs_and_std_devs["std_devs"][i]
        res.append(val)

    return res
