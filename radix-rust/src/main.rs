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

#[derive(Debug, PartialEq)]
enum FindResult {
    Found,
    NotFound,
}

fn _find_subpath_of(
    current: &Node,
    tokens: &[&str],
    token_path: &mut Vec<String>,
    variables: &mut HashMap<String, String>,
    exact: bool,
) -> FindResult {
    let is_leaf = current.token_to_child.is_none() && current.var_to_child.is_none();

    if is_leaf && !tokens.is_empty() && !exact {
        return FindResult::Found;
    }

    if tokens.is_empty() && current.is_valid_path {
        return FindResult::Found;
    }

    let [child_token, rest @ ..] = tokens else {
        return FindResult::NotFound;
    };

    if let Some(token_children) = &current.token_to_child {
        if let Some(node) = token_children.get(*child_token) {
            token_path.push(child_token.to_string());
            if _find_subpath_of(node, rest, token_path, variables, exact) == FindResult::Found {
                return FindResult::Found;
            } else {
                token_path.pop();
            }
        };
    }

    if let Some(var_children) = &current.var_to_child {
        token_path.push(child_token.to_string());
        let entry = var_children.iter().find(|(_, node)| {
            _find_subpath_of(node, rest, token_path, variables, exact) == FindResult::Found
        });

        if let Some((key, node)) = entry {
            variables.insert(key.to_owned(), child_token.to_string());
            return FindResult::Found;
        }
    }

    FindResult::NotFound
}

#[derive(Debug)]
pub struct Match {
    path: Vec<String>,
    variables: Option<HashMap<String, String>>,
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

    pub fn find(&self, path: &str) -> Option<Match> {
        let tokens: Vec<&str> = path.split('/').collect();
        let mut token_path: Vec<String> = Vec::new();
        let mut variables: HashMap<String, String> = HashMap::new();
        let result = _find_subpath_of(
            &self.root,
            tokens.as_slice(),
            &mut token_path,
            &mut variables,
            true,
        );

        if result == FindResult::Found {
            Some(Match {
                path: token_path,
                variables: if variables.is_empty() {
                    None
                } else {
                    Some(variables)
                },
            })
        } else {
            None
        }
    }
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("{:#?}", self.root))
    }
}

fn main() {
    let paths = vec![
        "a/b/{x}/c",
        "a/b/{x}/d",
        "a/b/{x}",
        "a/b/{y}/k",
        "a/b/{z}/v",
        "stuff/{firstName}/{lastName}",
        "stuff/{firstName}/{lastName}/id",
        "a/b/c/d",
        "a/b",
        "salt/pepper",
    ];

    let index = Index::from_paths(paths.as_slice());
    println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    let subpath = index.find("stuff/naphat/amundsen");
    println!("LOG:\x1b[33mDEBUG\x1b[0m: subpath: {:#?}", subpath);
}
