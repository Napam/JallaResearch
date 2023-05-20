#![allow(unused)]

use core::fmt;
use std::{collections::HashMap, vec};

#[derive(Default)]
struct Node {
    pub children: Option<HashMap<String, Node>>,
    pub is_valid_path: bool,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.children {
            None => write!(f, "<- leaf"),
            Some(children) => f
                .debug_map()
                .entries(children.iter().map(|(token, node)| {
                    if node.is_valid_path {
                        (token.to_owned() + "*", node)
                    } else {
                        (token.to_owned(), node)
                    }
                }))
                .finish(),
        }
    }
}

impl Node {
    fn new() -> Node {
        Node {
            children: None,
            is_valid_path: false,
        }
    }

    fn update(&mut self, tokens: &[&str]) {
        let Some(&token_of_child) = tokens.first() else { return };
        if token_of_child.is_empty() {
            return;
        };
        let mut map = self.children.get_or_insert_with(HashMap::new);
        let child = map
            .entry(token_of_child.to_string())
            .or_insert_with(Node::new);

        if tokens.len() == 1 {
            child.is_valid_path = true;
        }

        child.update(&tokens[1..]);
    }
}

struct Index {
    root: Node,
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("{:#?}", self.root))
    }
}

enum FindResult {
    Found,
    NotFound,
}

fn _find_subpath_of(current: &Node, tokens: &[&str], token_path: &mut Vec<String>) -> FindResult {
    if (tokens.is_empty() && current.is_valid_path) {
        return FindResult::Found;
    }

    let [token_of_child, rest @ ..] = tokens else {
        return current.children.as_ref().map_or(FindResult::Found, |_| FindResult::NotFound);
    };

    if let Some(children) = &current.children {
        token_path.push(token_of_child.to_string());
        children
            .get(&token_of_child.to_string())
            .map_or(FindResult::NotFound, |node| {
                _find_subpath_of(node, rest, token_path)
            })
    } else {
        FindResult::Found
    }
}

impl Index {
    fn from(paths: &[&str]) -> Self {
        let mut map: HashMap<String, Node> = HashMap::new();
        let mut root = Node::new();

        for path in paths.iter() {
            let path = path.trim_matches('/');
            let tokens = path.split('/').collect::<Vec<&str>>();
            root.update(tokens.as_slice());
        }

        Index { root }
    }

    fn find_subpath_of(&self, path: &str) -> Option<Vec<String>> {
        let tokens: Vec<&str> = path.split('/').collect();
        let mut token_path: Vec<String> = Vec::new();
        match _find_subpath_of(&self.root, tokens.as_slice(), &mut token_path) {
            FindResult::Found => Some(token_path),
            FindResult::NotFound => None,
        }
    }
}

fn main() {
    let paths = vec!["a/b/c", "a/r", "a/r/f/d", "b/c", "c", ""];

    let index = Index::from(paths.as_slice());
    println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    let subpath = index.find_subpath_of("a/r/f/d/e");
    println!("LOG:\x1b[33mDEBUG\x1b[0m: subpath: {:?}", subpath);
}
