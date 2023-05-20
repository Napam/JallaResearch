#![allow(unused)]

use core::fmt;
use std::{collections::HashMap, vec};

#[derive(Default)]
struct Node {
    pub children: Option<HashMap<String, Node>>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.children {
            None => write!(f, "leaf"),
            Some(children) => f.debug_map().entries(children.iter()).finish(),
        }
    }
}

impl Node {
    fn new() -> Node {
        Node { children: None }
    }

    fn update(&mut self, tokens: &[&str]) {
        let Some(&first) = tokens.first() else { return };
        let mut map = self.children.get_or_insert_with(HashMap::new);
        let child = map.entry(first.to_string()).or_insert_with(Node::new);
        child.update(&tokens[1..]);
    }
}

struct Index {
    map: HashMap<String, Node>,
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.map.iter()).finish()
    }
}

enum FindResult {
    Found,
    NotFound,
}

fn _find_subpath_of(current: &Node, tokens: &[&str], cum: &mut Vec<String>) -> FindResult {
    let [token_of_child, rest @ ..] = tokens else {
        return current.children.as_ref().map_or(FindResult::Found, |_| FindResult::NotFound);
    };

    if let Some(children) = &current.children {
        cum.push(token_of_child.to_string());
        children.get(&token_of_child.to_string())
                .map_or(FindResult::NotFound, |node| _find_subpath_of(node, rest, cum))
    } else {
        FindResult::Found
    }
}

impl Index {
    fn from(paths: &[&str]) -> Self {
        let mut map: HashMap<String, Node> = HashMap::new();

        for path in paths.iter() {
            let path = path.trim_matches('/');
            let tokens = path.split('/').collect::<Vec<&str>>();
            let [first, rest @ ..] = tokens.as_slice() else { continue };
            if first.is_empty() {
                continue;
            }

            let mut node = map.entry(first.to_string()).or_insert_with(Node::new);
            node.update(rest);
        }

        Index { map }
    }

    fn find_subpath_of(&self, path: &str) -> Option<String> {
        let tokens: Vec<&str> = path.split('/').collect();
        let [token_of_child, rest @ ..] = tokens.as_slice() else { return None };

        match self.map.get(&token_of_child.to_string()) {
            None => None,
            Some(node) => {
                let mut cum = vec![token_of_child.to_string()];
                match _find_subpath_of(node, rest, &mut cum) {
                    FindResult::Found => Some(cum.join("/")),
                    FindResult::NotFound => None
                }
            }
        }
    }
}

fn main() {
    let paths = vec!["a/b/c", "a/r/f", "a/r/f/d", "b/c", "c", ""];

    let index = Index::from(paths.as_slice());
    println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    let subpath = index.find_subpath_of("b/c");
    println!("LOG:\x1b[33mDEBUG\x1b[0m: subpath: {:?}", subpath);
}
