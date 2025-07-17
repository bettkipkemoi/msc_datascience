class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  # Height of the node (leaf nodes have height 1)

class AVLTree:
    def __init__(self):
        self.root = None

    def height(self, node):
        """Return the height of a node (0 if None)."""
        return node.height if node else 0

    def balance_factor(self, node):
        """Calculate balance factor (height of left - height of right)."""
        return self.height(node.left) - self.height(node.right) if node else 0

    def update_height(self, node):
        """Update the height of a node based on children."""
        if node:
            node.height = max(self.height(node.left), self.height(node.right)) + 1

    def right_rotate(self, y):
        """Perform a right rotation on node y."""
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self.update_height(y)
        self.update_height(x)
        return x

    def left_rotate(self, x):
        """Perform a left rotation on node x."""
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self.update_height(x)
        self.update_height(y)
        return y

    def insert(self, value):
        """Insert a value into the AVL tree and balance it."""
        self.root = self._insert(self.root, value)

    def _insert(self, node, value):
        # Standard BST insertion
        if not node:
            return AVLNode(value)
        if value < node.value:
            node.left = self._insert(node.left, value)
        elif value > node.value:
            node.right = self._insert(node.right, value)
        else:
            return node  # Duplicate values not allowed

        # Update height of current node
        self.update_height(node)

        # Check balance factor
        balance = self.balance_factor(node)

        # Left Left Case (LL)
        if balance > 1 and value < node.left.value:
            return self.right_rotate(node)

        # Right Right Case (RR)
        if balance < -1 and value > node.right.value:
            return self.left_rotate(node)

        # Left Right Case (LR)
        if balance > 1 and value > node.left.value:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Right Left Case (RL)
        if balance < -1 and value < node.right.value:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def search(self, value):
        """Search for a value in the AVL tree."""
        return self._search(self.root, value)

    def _search(self, node, value):
        if not node or node.value == value:
            return node
        if value < node.value:
            return self._search(node.left, value)
        return self._search(node.right, value)

    def inorder_traversal(self):
        """Perform an inorder traversal to print values in sorted order."""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.value)
            self._inorder(node.right, result)

# Example usage
def main():
    avl = AVLTree()
    values = [10, 20, 30, 40, 50, 25]
    for value in values:
        avl.insert(value)
    
    print("Inorder Traversal of AVL Tree:", avl.inorder_traversal())
    
    # Search for a value
    search_value = 25
    result = avl.search(search_value)
    print(f"Search for {search_value}: {'Found' if result else 'Not Found'}")

if __name__ == "__main__":
    main()