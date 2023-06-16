#![allow(unused)]

mod simpleindex;
use core::fmt;

use std::{collections::HashMap, format};

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
            .map(|(k, v)| (k.to_owned(), v))
            .chain(
                self.var_to_child
                    .iter()
                    .flatten()
                    .map(|(k, v)| (format!("{{{}}}", k), v)),
            );

        if self.is_leaf() {
            write!(f, "")
        } else {
            f.debug_map()
                .entries(iter.map(|(token, node)| {
                    if node.is_valid_path {
                        (token + "*", node)
                    } else {
                        (token, node)
                    }
                }))
                .finish()
        }
    }
}

fn is_variable(token: &str) -> bool {
    token.starts_with('{') && token.ends_with('}')
}

#[derive(Debug)]
pub struct Token<T> {
    value: T,
    is_variable: bool,
}

impl<T> Token<T> {
    fn new(value: T, is_variable: bool) -> Self {
        Token { value, is_variable }
    }

    fn literal(value: T) -> Self {
        Token {
            value,
            is_variable: false,
        }
    }

    fn variable(value: T) -> Self {
        Token {
            value,
            is_variable: true,
        }
    }
}

impl Node {
    fn new() -> Node {
        Node {
            token_to_child: None,
            var_to_child: None,
            is_valid_path: false,
        }
    }

    fn is_leaf(&self) -> bool {
        self.token_to_child.is_none() && self.var_to_child.is_none()
    }

    fn update(&mut self, tokens: &[Token<&str>]) {
        let [token_of_child, rest @ .. ] = tokens else {
            self.is_valid_path = true;
            return
        };

        let map = if token_of_child.is_variable {
            self.var_to_child.get_or_insert_with(HashMap::new)
        } else {
            self.token_to_child.get_or_insert_with(HashMap::new)
        };

        let child = map
            .entry(token_of_child.value.to_string())
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
    if current.is_leaf() && !tokens.is_empty() && !exact {
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
    pub fn from_url_paths(paths: &[&str]) -> Self {
        let mut root = Node::new();

        for path in paths.iter() {
            let path = path.trim_matches('/');
            let tokens = path.split('/').collect::<Vec<&str>>();
            root.update(
                tokens
                    .iter()
                    .map(|token| {
                        if is_variable(token) {
                            Token::variable(token.to_owned().trim_matches(|x| x == '{' || x == '}'))
                        } else {
                            Token::literal(token.to_owned())
                        }
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
        }

        Index { root }
    }

    // pub fn from_token_paths(paths: &Vec<Vec<Token<&str>>>) -> Self {
    //     let mut root = Node::new();

    //     for path in paths.iter() {
    //         root.update(tokens.as_slice());
    //     }

    //     Index { root }
    // }

    pub fn from_paths() {}

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
    let url_paths = vec![
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

    let token_paths = vec![
        vec![
            Token::literal("a"),
            Token::literal("b"),
            Token::variable("x"),
            Token::literal("c"),
        ],
        vec![
            Token::literal("a"),
            Token::literal("b"),
            Token::variable("x"),
            Token::literal("d"),
        ],
        vec![
            Token::literal("a"),
            Token::literal("b"),
            Token::variable("x"),
        ],
        vec![Token::literal("salt"), Token::literal("pepper")],
    ];

    // let index = Index::from_token_paths(&token_paths);
    // println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    let index = Index::from_url_paths(url_paths.as_slice());
    println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    // let subpath = index.find("stuff/naphat/amundsen");
    // println!("LOG:\x1b[33mDEBUG\x1b[0m: subpath: {:#?}", subpath);
}
