import math as mt


class TreeNode:
    val = 0
    left = None
    right = None
    is_accessed = False

    def __init__(self, val):
        self.val = val


def get_balanced_kdtree(train_dataset, depth=0):
    if len(train_dataset) == 0:
        return None

    cur_dim = depth % (len(train_dataset[0]) - 1)
    mid_index = len(train_dataset) >> 1

    sorted_list = train_dataset.copy()
    sorted_list.sort(key=lambda item: item[cur_dim])

    root = TreeNode(sorted_list[mid_index])
    root.left = get_balanced_kdtree(sorted_list[:mid_index], depth + 1)
    root.right = get_balanced_kdtree(sorted_list[mid_index + 1:], depth + 1)

    return root


def get_knn(balanced_kdtree, k, sample, parent=None, distances=[], depth=0):
    if balanced_kdtree is None:
        return None

    length_of_sample = len(sample)
    balanced_kdtree.is_accessed = True
    cur_dim = depth % len(sample)

    if sample[cur_dim] <= balanced_kdtree.val[cur_dim]:
        get_knn(balanced_kdtree.left, k, sample, balanced_kdtree, distances, depth + 1)
    else:
        get_knn(balanced_kdtree.right, k, sample, balanced_kdtree, distances, depth + 1)

    cur_distance = 0

    for i in range(length_of_sample):
        cur_distance += (sample[i] - balanced_kdtree.val[i]) ** 2

    cur_distance = mt.sqrt(cur_distance)
    distances.append([cur_distance, balanced_kdtree.val[-1]])

    if parent is not None and parent.left is not None and parent.right is not None:
        another_distance = (sample[cur_dim - 1] - parent.val[cur_dim - 1]) ** 2

        if another_distance < cur_distance:
            if parent.left is balanced_kdtree and parent.right.is_accessed == False:
                get_knn(balanced_kdtree.right, k, sample, parent, distances, depth)
            elif parent.right is balanced_kdtree and parent.left.is_accessed == False:
                get_knn(balanced_kdtree.left, k, sample, parent, distances, depth)

    knn = sorted(distances, key=lambda distance: distance[0])

    return knn[:k]


class Classifier:
    balanced_kdtree = None
    k = 0

    def __init__(self, balanced_kdtree, k):
        self.balanced_kdtree = balanced_kdtree
        self.k = k

    def classify(self, sample):
        knn = get_knn(self.balanced_kdtree, self.k, sample)
        record = {}

        for e in knn:
            if e[1] in record:
                record[e[1]] += 1 / e[0]
            else:
                record[e[1]] = 1 / e[0]

        items = list(record.items())
        items.sort(key=lambda item: item[1])

        prob = 0

        for v in record.values():
            prob += v

        prob = items[-1][1] / prob

        for i in range(len(knn)):
            knn[i] = knn[i][1]

        res = {
            "type": items[-1][0],
            "knn": knn,
            "prob": prob,
        }

        return res


def test():
    train_dataset = [[1, 1, 1, 1, '女神'],
                     [0.5, 1, 1, 1, '淑女'],
                     [0.1, 0.1, 0.1, 0.1, '丑女'],
                     [0.5, 0.5, 0.5, 0.5, '一般型'],
                     [1, 0.8, 0.3, 1, '淑女'],
                     [0.6, 0.5, 0.7, 0.5, '一般型'],
                     [1, 1, 0.9, 0.5, '女神'],
                     [1, 0.6, 0.5, 0.8, '一般型'],
                     [0.5, 0.5, 1, 1, '淑女'],
                     [0.9, 1, 1, 1, '女神'],
                     [0.6, 0.6, 1, 0.1, '丑女'],
                     [1, 0.8, 0.5, 0.5, '可爱型'],
                     [1, 0.1, 0.1, 1, '可爱型'],
                     [1, 1, 0.7, 0.3, '淑女'],
                     [0.2, 0.3, 0.4, 0.5, '丑女'],
                     [0.5, 1, 0.6, 0.6, '可爱型']]

    sample = [1, 1, 0.9, 1]

    balanced_kdtree_1 = get_balanced_kdtree(train_dataset)
    cls_1 = Classifier(balanced_kdtree_1, 3)
    res = cls_1.classify(sample)

    print(res)


test()




