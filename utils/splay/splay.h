#ifndef _SPLAY_H_
#define _SPLAY_H_

#include <iostream>

class SplayTree {
 private:
 	class Node {
 		public:
 			Node(int key) : key(key), left(NULL), right(NULL) {}
 			int key;
 			Node *left;
 			Node *right;
 	};
 
 	Node *_root;
 
 public:
  SplayTree() : _root(NULL) {}

 	bool insert(int key) {
 		Node *new_node = new Node(key);
 		if (this->_root == NULL) {
 			this->_root = new_node;
 		} else {
 			Node* found = splay(_root, key);
      this->_root = found;
 			if (key < found->key) {
 				new_node->left = found->left;
 				new_node->right = found;
 				found->left = NULL;
        this->_root = new_node;
 			} else if (key > found->key) {
 				new_node->right = found->right;
 				new_node->left = found;
        found->right = NULL;
        this->_root = new_node;
 			} else {
 				return false;
 			}
 		}
 		return true;
 	}

	bool remove(int key) {
 		Node *found = splay(_root, key);
    this->_root = found;
    if (found->key != key) {
      return false;
		}
		if (found->left == NULL) {
			this->_root = found->right;
		} else {
			Node *left = splay(found->left, key);
			left->right = found->right;
			this->_root = left;
		}
		return true;
	}

  bool lookup(int key) {
 		Node *found = splay(_root, key);
    this->_root = found;
    if (found->key != key) {
      return false;
		} else {
      return true;
    }
  }

	void traverse() {
		if (this->_root) {
			recursive_display(this->_root);	
		}
	}
 
 private:
 	Node *splay(Node *root, int key) {
 		Node *dummy_node = new Node(0);
 		Node *ltree_max = NULL, *rtree_min = NULL, *yy = NULL;
		ltree_max = rtree_min = dummy_node;
 		while (true) {
 			if (key < root->key) {
 				if ((yy = (root->left)) == NULL) {
 					break;
 				}
 				if (key < yy->key) {
 					root->left = yy->right;
 					yy->right = root;
 					root = yy;
 					if ((yy = (root->left)) == NULL) {
 						break;
 					}
 				}
 				rtree_min->left = root;
 				rtree_min = root;
 			} else if (key > root->key) {
 				if ((yy = (root->right)) == NULL) {
 					break;
 				}
 				if (key > yy->key) {
 					root->right = yy->left;
 					yy->left = root;
 					root = yy;
 					if ((yy = (root->right)) == NULL) {
 						break;
 					}
 				}
 				ltree_max->right = root;
 				ltree_max = root;
 			} else {
 				break;
 			}
 			root = yy;
 		}
 		ltree_max->right = root->left;
 		rtree_min->left = root->right;
		root->left = dummy_node->right;
    root->right = dummy_node->left;
 	}

	void recursive_display(Node *root) {
		std::cout << "root: " << root->key << std::endl;
		if (root->left) {
			std::cout << "left: " << root->left->key << std::endl;
		}
		if (root->right) {
			std::cout << "right: " << root->right->key << std::endl;
		}
		if (root->left) {
			recursive_display(root->left);
		}
		if (root->right) {
			recursive_display(root->right);
		}
	}
};

#endif
