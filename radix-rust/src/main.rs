#![allow(unused)]

mod simpleindex;
use core::fmt;

use std::collections::HashMap;

pub struct Index {
    root: Node,
}

#[derive(Default)]
pub struct Node {
    token_to_child: Option<HashMap<String, Node>>,
    var_to_child: Option<HashMap<String, Node>>,
    is_valid_path: bool,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self
            .token_to_child
            .iter()
            .flatten()
            .chain(self.var_to_child.iter().flatten());

        f.debug_map()
            .entries(iter.map(|(token, node)| {
                if node.is_valid_path {
                    (token.to_owned() + "*", node)
                } else {
                    (token.to_owned(), node)
                }
            }))
            .finish()
    }
}

fn is_variable(token: &str) -> bool {
    token.starts_with('{') && token.ends_with('}')
}

impl Node {
    fn new() -> Node {
        Node {
            token_to_child: None,
            var_to_child: None,
            is_valid_path: false,
        }
    }

    fn update(&mut self, tokens: &[&str]) {
        let [token_of_child, rest @ .. ] = tokens else {
            self.is_valid_path = true;
            return
        };

        let map = if is_variable(token_of_child) {
            self.var_to_child.get_or_insert_with(HashMap::new)
        } else {
            self.token_to_child.get_or_insert_with(HashMap::new)
        };

        let child = map
            .entry(token_of_child.to_string())
            .or_insert_with(Node::new);

        child.update(rest);
    }
}

impl Index {
    pub fn from_paths(paths: &[&str]) -> Self {
        let mut root = Node::new();

        for path in paths.iter() {
            let path = path.trim_matches('/');
            let tokens = path.split('/').collect::<Vec<&str>>();
            root.update(tokens.as_slice());
        }

        Index { root }
    }
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("{:#?}", self.root))
    }
}

fn run_varindex() {
    let paths = vec![
        "a/b/{thing}/c",
        "a/b/{thing}/d",
        "a/b/{thing}",
        "a/b/{thang}/k",
        "a/b/c/d",
        "a/b",
    ];

    let index = Index::from_paths(paths.as_slice());
    println!("log:\x1b[33mdebug\x1b[0m: index: {:#?}", index);

    // "a/b/w/k";
    // let subpath = index.find_subpath_of("a/b/{thing}/c");
    // println!("log:\x1b[33mdebug\x1b[0m: subpath: {:?}", subpath);
}

fn run_simpleindex() {
    // let paths = vec!["a/b/c", "a/r", "a/r/f/d", "b/c", "c", ""];
    let paths = vec!["a/b/{thing}/c"];

    let index = simpleindex::Index::from_paths(paths.as_slice());
    println!("log:\x1b[33mdebug\x1b[0m: index: {:#?}", index);

    let subpath = index.find_subpath_of("a/b/{thing}/c");
    println!("log:\x1b[33mdebug\x1b[0m: subpath: {:?}", subpath);
}

fn main() {
    // run_simpleindex();
    run_varindex();
}
