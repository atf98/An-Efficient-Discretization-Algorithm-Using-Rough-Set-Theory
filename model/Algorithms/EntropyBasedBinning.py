from scipy.stats import entropy

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None



class binary_tree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        #判斷tree是否為空
        if self.root == None:
            self.root = Node(data)
        else:
            self._insert(data, self.root)
    
    def _insert(self, data, cur_node):
        if data[0][0]+data[-1][0] < cur_node.data[0][0]+cur_node.data[-1][0]:
            if cur_node.left == None:
                cur_node.left = Node(data)
            else:
                self._insert(data, cur_node.left)
        elif data[0][0]+data[-1][0] > cur_node.data[0][0]+cur_node.data[-1][0]:
            if cur_node.right == None:
                cur_node.right = Node(data)
            else:
                self._insert(data, cur_node.right)

# Function to get leafnodes from left to right
def getLeafNodes(root, LeafNodes):

    # If node is null, return
    if (not root):
        return

    # If node is leaf node, print its data
    if root.left==None and root.right==None:
        LeafNodes.append(root.data)
        # print(root.data)

    # If left child exists, check for leaf recursively
    if root.left:
        getLeafNodes(root.left,LeafNodes)
 
    # If right child exists, check for leaf recursively
    if root.right:
        getLeafNodes(root.right,LeafNodes)

# Function to insert children intp leaf
def insertLeafNodes(root, left_child, right_child):

    # If node is null, return
    if (not root):
        return

    # If node is leaf node, print its data
    if root.data[0][0] == left_child[0][0] and root.data[-1][0] == right_child[-1][0]:
        root.left = Node(left_child)
        root.right = Node(right_child)
        return

    # If left child exists, check for leaf recursively
    if root.left:
        insertLeafNodes(root.left, left_child, right_child)
 
    # If right child exists, check for leaf recursively
    if root.right:
        insertLeafNodes(root.right, left_child, right_child)

# get bin dict
def get_bin(LeafNodes):
    bin_dict = {}
    bins = [0]
    for leaf in LeafNodes:
        start = leaf[0][0]
        end = leaf[-1][0]
        bin = ("{}-{}".format(start,end))
        bin_dict[bin] = [start,end]
        bins += [end]
    return bin_dict, bins

# calculate probability
def cal_prob(a_list):
    a_len = len(a_list)
    a_sum = 0
    for i in a_list:
        a_sum += i[1]
    a_p1 = a_sum/a_len
    a_p0 = 1-a_p1
    return a_len, a_p1, a_p0


def cal_entropy(df,variable, class_label_name):
  
    df1 = df.sort_values(variable)
    independent_list = df1[variable].tolist()
    response_list = df1[class_label_name].tolist()

    total_len = len(independent_list)
    # print(total_len)
    pair = []
    for i in range(total_len):
        pair.append([independent_list[i],response_list[i]])
    # print(pair)

    tree = binary_tree()
    tree.insert(pair)
    # print(tree.root.data)

    p1 = sum(response_list)/total_len
    p0 = 1-p1
    original_entropy = entropy([p1,p0], base=2)
    entropy_list = [original_entropy]
    # print(original_entropy)

    LeafNodes = [pair]
    # print(LeafNodes)

    k = 1
    while True:
        information_gained = 0

        for leaf in LeafNodes:
            inside_information_gained = 0

            if len(leaf) == 1:
              continue

            # get inside breakpoints
            breakpoints = []
            for i in range(len(leaf)-1):
                if leaf[i][0] != leaf[i+1][0]:
                    breakpoints.append(i+1)
            # print(breakpoints)

            if breakpoints == []:
              continue

            # initialize information gained
            
            total_len_0, p1, p0 =cal_prob(leaf)
            # print(total_len_0, p1, p0)
            inside_entropy = total_len_0/total_len*entropy([p1,p0], base=2)
            # print(inside_entropy)

            for breakpoint in breakpoints:
                leaf_1 = leaf[:breakpoint]
                # print(leaf_1)
                leaf_2 = leaf[breakpoint:]
                # print(leaf_2)
                total_len_1, p1_1, p1_0 = cal_prob(leaf_1)
                # print(total_len_1, p1_1, p1_0)
                total_len_2, p2_1, p2_0 = cal_prob(leaf_2)
                # print(total_len_2, p2_1, p2_0)
                inside_new_entropy = total_len_1/total_len*entropy([p1_1,p1_0],base=2) + total_len_2/total_len*entropy([p2_1,p2_0],base=2)
                # print(inside_new_entropy)
                inside_information_gained = inside_entropy - inside_new_entropy
                # print(inside_information_gained)
                if inside_information_gained > information_gained:
                    information_gained = inside_information_gained
                    # print(information_gained)
                    temp_leaf_1 = leaf_1
                    # print(temp_leaf_1)
                    temp_leaf_2 = leaf_2
                    # print(temp_leaf_2)
        
        # run to the end
        if information_gained == 0:
          break

        #threshold: information_gained
        # if information_gained < 0.01:
        #   break
 

        insertLeafNodes(root=tree.root, left_child=temp_leaf_1, right_child=temp_leaf_2)
        original_entropy -= information_gained
        # print(original_entropy)
        entropy_list.append(original_entropy)
        # print(entropy_list)
        LeafNodes=[]
        getLeafNodes(tree.root,LeafNodes)
        # print(LeafNodes)
        # print("finish round", k, end =" : ")
        # print_result(LeafNodes)
        k += 1

        #threshold: run times
        if k == 4:
          part_LeafNodes = LeafNodes.copy()
          part_entropy_list = entropy_list.copy()
          # break 


    # print(entropy_list)
    return part_LeafNodes, part_entropy_list, LeafNodes, entropy_list