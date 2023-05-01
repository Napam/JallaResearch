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

    fn update(&mut self, tokens: &Vec<&str>) {
        let Some(&first) = tokens.first() else { return };
        let mut map = self.children.get_or_insert_with(|| HashMap::new());
        let child = map.entry(first.to_string()).or_insert(Node::new());
        child.update(&tokens[1..].to_vec());
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

impl Index {
    fn from(paths: &Vec<String>) -> Self {
        let mut map: HashMap<String, Node> = HashMap::new();

        for path in paths.iter() {
            let path = path.trim_matches('/');
            let tokens = path.split('/').collect::<Vec<&str>>();
            let [first, rest @ ..] = tokens.as_slice() else { continue };
            if first.is_empty() {
                continue;
            }

            let mut node = map.entry(first.to_string()).or_insert_with(|| Node::new());
            node.update(&rest.to_vec());
        }

        Index { map }
    }

    fn _find_subpath_of(&self, current: &Node, tokens: &[&str]) -> Option<String> {
        let [first, rest @ ..] = tokens else { return None };
        match &current.children {
            None => None,
            Some(map) => map
                .get(&first.to_string())
                .and_then(|node| self._find_subpath_of(node, rest))
                .map(|_| "aaa".to_string()),
        }
    }

    fn find_subpath_of(&self, path: &String) -> Option<String> {
        let tokens: Vec<&str> = path.split("/").collect();
        let [first, rest @ ..] = tokens.as_slice() else { return None };

        match self.map.get(&first.to_string()) {
            None => None,
            Some(node) => self._find_subpath_of(node, rest)
        }
    }
}

fn main() {
    let paths = vec![
        "a/b/c".to_string(),
        "a/r/f".to_string(),
        "a/r/f/d".to_string(),
        "b/c".to_string(),
        "c".to_string(),
        "".to_string(),
    ];

    let index = Index::from(&paths);
    println!("{:#?}", index);

    let sub = index.find_subpath_of(&"b/c".to_string());
    // index.find_subpath_of(&"x/b/c/d/e".to_string());
    println!("{:?}", sub);
}
