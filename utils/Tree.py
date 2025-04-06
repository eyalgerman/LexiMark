import json

class TreeNode:
    """
    A class representing a node in a tree structure used for tracking word substitutions.

    Each node represents a substitution pair (original → replaced) and may have children representing
    further substitutions or derivations. The tree can be serialized, deserialized, and printed.

    Attributes:
        original (str): The original word or phrase.
        replaced (str): The replacement or synonym.
        children (List[TreeNode]): List of child nodes.
        model_prediction (float): Optional score or prediction associated with this substitution.
    """
    def __init__(self, original, replaced, model_prediction=0):
        """
        Initializes a TreeNode with given original and replaced words.

        Args:
            original (str): The original word or phrase.
            replaced (str): The replaced or substituted version.
            model_prediction (float, optional): A prediction score (e.g., model confidence). Defaults to 0.
        """
        self.original = original
        self.replaced = replaced
        self.children = []
        self.model_prediction = model_prediction

    def add_child(self, child_node):
        """
        Adds a child node to the current node if it doesn't already exist.

        Args:
            child_node (TreeNode): The child node to be added.

        Returns:
            TreeNode: The added or existing child node.
        """
        for child in self.children:
            if child.original == child_node.original:
                return child
        self.children.append(child_node)
        return child_node

    def __repr__(self):
        """
        Returns a string representation of the TreeNode.

        Returns:
            str: A string showing the original and replaced values.
        """
        return f"TreeNode({self.original} : {self.replaced})"

    def update_replaced(self, original, new_replaced):
        """
        Updates the replaced value of the node that matches the given original word.

        Args:
            original (str): The original word to search for.
            new_replaced (str): The new replacement word.

        Returns:
            bool: True if the replacement was updated, False otherwise.
        """
        node = self.search_by_original(original)
        if node:
            node.replaced = new_replaced
            return True
        return False

    def search_by_original(self, original):
        """
        Recursively searches the tree for a node with the specified original word.

        Args:
            original (str): The original word to search for.

        Returns:
            TreeNode or None: The matching node, or None if not found.
        """
        if self.original == original:
            return self
        for child in self.children:
            result = child.search_by_original(original)
            if result:
                return result
        return None

    def search_by_replaced(self, replaced):
        """
        Recursively searches the tree for a node with the specified replaced word.

        Args:
            replaced (str): The replaced word to search for.

        Returns:
            TreeNode or None: The matching node, or None if not found.
        """
        if self.replaced == replaced:
            return self
        for child in self.children:
            result = child.search_by_replaced(replaced)
            if result:
                return result
        return None

    def search_by_replaced_one_level(self, replaced):
        """
        Searches only the current node's children for a node with the specified replaced word.

        Args:
            replaced (str): The replaced word to search for.

        Returns:
            TreeNode or None: The matching child node, or None if not found.
        """
        for child in self.children:
            if child.replaced == replaced:
                return child
        return None

    def add_child_to_original(self, original, child_node):
        """
        Adds a child to a node identified by the given original word.

        Args:
            original (str): The original word to locate the parent node.
            child_node (TreeNode): The child node to add.

        Returns:
            bool: True if added successfully, False otherwise.
        """
        parent_node = self.search_by_original(original)
        if parent_node:
            parent_node.add_child(child_node)
            return True
        return False

    def to_dict(self):
        """
        Converts the tree starting from this node into a dictionary format.

        Returns:
            dict: A dictionary representing the node and its children.
        """
        return {
            'original': self.original,
            'replaced': self.replaced,
            'model_prediction': self.model_prediction,
            'children': [child.to_dict() for child in self.children]

        }

    @classmethod
    def from_dict(cls, data):
        """
        Constructs a TreeNode (and its children recursively) from a dictionary.

        Args:
            data (dict): Dictionary representing the tree structure.

        Returns:
            TreeNode: The root node of the reconstructed tree.
        """
        node = cls(data['original'], data['replaced'])
        for child_data in data['children']:
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)
        return node

    def save_to_file(self, file_path):
        """
        Saves the tree structure to a JSON file.

        Args:
            file_path (str): Path to the file where the tree should be saved.
        """
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load_from_file(cls, file_path):
        """
        Loads a tree structure from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            TreeNode: The root node of the loaded tree.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
            return cls.from_dict(data)

    def print_tree(self, level=0):
        """
        Recursively prints the tree in a readable format, showing hierarchy.

        Args:
            level (int, optional): Current depth level for indentation. Defaults to 0.
        """
        indent = "  " * level
        print(f"{indent}{self.original} : {self.replaced}")
        for child in self.children:
            child.print_tree(level + 1)



