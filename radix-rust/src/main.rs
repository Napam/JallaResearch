#![allow(unused)]

mod simpleindex;
use core::fmt;

use std::{collections::HashMap, fmt::Debug, format};

pub struct RadixMap<T> {
    root: Node<T>,
}

#[derive(Default)]
pub struct Node<T> {
    token_to_child: Option<HashMap<String, Node<T>>>,
    var_to_child: Option<HashMap<String, Node<T>>>,
    is_valid_path: bool,
    value: Option<T>,
}

impl<T: std::fmt::Debug> fmt::Debug for Node<T> {
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
                .entries(
                    iter.map(|(token, node)| match (node.is_valid_path, &node.value) {
                        (true, Some(value)) => (token + "*" + &format!(" ({:?})", value), node),
                        (true, None) => (token + "*" + "(None)", node),
                        (false, _) => (token, node),
                    }),
                )
                .finish()
        }
    }
}

fn is_variable(token: &str) -> bool {
    token.starts_with('{') && token.ends_with('}')
}

#[derive(Debug)]
pub struct Token {
    value: String,
    is_variable: bool,
}

impl Token {
    fn new(value: &str, is_variable: bool) -> Self {
        Token {
            value: value.to_string(),
            is_variable,
        }
    }

    fn literal(value: &str) -> Self {
        Token {
            value: value.to_string(),
            is_variable: false,
        }
    }

    fn variable(value: &str) -> Self {
        Token {
            value: value.to_string(),
            is_variable: true,
        }
    }
}

impl<T> Node<T> {
    fn new(value: Option<T>) -> Node<T> {
        Node {
            token_to_child: None,
            var_to_child: None,
            is_valid_path: false,
            value,
        }
    }

    fn is_leaf(&self) -> bool {
        self.token_to_child.is_none() && self.var_to_child.is_none()
    }

    fn update(&mut self, tokens_and_value: (&[Token], T)) {
        let [token_of_child, rest @ .. ] = tokens_and_value.0 else {
            self.is_valid_path = true;
            self.value = Some(tokens_and_value.1);
            return
        };

        let map = if token_of_child.is_variable {
            self.var_to_child.get_or_insert_with(HashMap::new)
        } else {
            self.token_to_child.get_or_insert_with(HashMap::new)
        };

        let child = map
            .entry(token_of_child.value.to_string())
            .or_insert_with(|| Node::new(None));

        child.update((rest, tokens_and_value.1));
    }
}

#[derive(Debug, PartialEq)]
enum FindResult {
    Found,
    NotFound,
}

fn _find_subpath_of<T>(
    current: &Node<T>,
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

impl<T: Copy> RadixMap<T> {
    pub fn from_url_paths_and_values(paths_and_values: &[(&str, T)]) -> Self {
        let mut root: Node<T> = Node::new(None);

        for (path, value) in paths_and_values.iter() {
            let path = path.trim_matches('/');
            let tokens = path.split('/').collect::<Vec<&str>>();
            root.update((
                tokens
                    .iter()
                    .map(|token| {
                        if is_variable(token) {
                            Token::variable(token.trim_matches(|x| x == '{' || x == '}'))
                        } else {
                            Token::literal(token.to_owned())
                        }
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
                *value,
            ));
        }

        RadixMap { root }
    }

    pub fn from_token_paths(paths_and_values: &[(Vec<Token>, T)]) -> Self {
        let mut root: Node<T> = Node::new(None);

        for (path, value) in paths_and_values {
            root.update((path.as_slice(), *value));
        }

        RadixMap { root }
    }

    pub fn from_paths() {}

    pub fn find_url(&self, path: &str) -> Option<Match> {
        let tokens: Vec<&str> = path.split('/').collect();
        self.get(tokens.as_slice())
    }

    pub fn get(&self, tokens: &[&str]) -> Option<Match> {
        let mut token_path: Vec<String> = Vec::new();
        let mut variables: HashMap<String, String> = HashMap::new();
        let result = _find_subpath_of(&self.root, tokens, &mut token_path, &mut variables, true);

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

impl<T: std::fmt::Debug> fmt::Debug for RadixMap<T> {
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

    let token_paths = [
        (
            vec![
                Token::literal("a"),
                Token::literal("b"),
                Token::variable("x"),
                Token::literal("c"),
            ],
            1,
        ),
        (vec![Token::literal("a"), Token::literal("b")], 2),
        (vec![Token::literal("salt"), Token::literal("pepper")], 3),
    ];

    let index = RadixMap::from_token_paths(&token_paths);
    println!("LOG:\x1b[33mDEBUG\x1b[0m: index: {:#?}", index);

    let subpath = index.find_url("a/b/asdf/c");
    println!("LOG:\x1b[33mDEBUG\x1b[0m: subpath: {:#?}", subpath);
}
