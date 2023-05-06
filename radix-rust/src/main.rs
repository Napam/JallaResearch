#![allow(unused)]

use core::fmt;
use std::collections::HashMap;

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

fn _find_subpath_of(current: &Node, tokens: &[&str]) -> Option<String> {
    let [token_of_child, rest @ ..] = tokens else { return None };

    current.children.as_ref().map(|map| {
        map.get(&token_of_child.to_string())
            .and_then(|node| _find_subpath_of(node, rest))
            .map_or(token_of_child.to_string(), |_| "aaa".to_string())
    })
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
        let [first, rest @ ..] = tokens.as_slice() else { return None };

        match self.map.get(&first.to_string()) {
            None => None,
            Some(node) => _find_subpath_of(node, rest),
        }
    }
}

fn main() {
    let paths = vec!["a/b/c", "a/r/f", "a/r/f/d", "b/c", "c", ""];

    let index = Index::from(paths.as_slice());
    println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    let sub = index.find_subpath_of("a/r/f/d/asdf");
    // index.find_subpath_of(&"x/b/c/d/e".to_string());
    println!("LOG:\x1b[33mDEBUG\x1b[0m: sub: {:?}", sub);
}
