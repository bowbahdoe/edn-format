use bigdecimal::{BigDecimal, ParseBigDecimalError};
use chrono::FixedOffset;
use internship::IStr;
use itertools::Itertools;
use num_bigint::{BigInt, ParseBigIntError};
use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::iter::FromIterator;
use std::num::{ParseFloatError, ParseIntError};
use thiserror::Error;
use uuid::Uuid;

/// A keyword, as described in EDN data model is identifier which should
/// "designate itself".
///
/// Because its contents are interned, cloning and comparisons should be relatively
/// cheap operations, while construction is relatively not.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Keyword {
    namespace: Option<IStr>,
    name: IStr,
}

impl Keyword {
    /// The namespace of the keyword, if there is one
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_ref().map(|s| s.as_str())
    }

    /// The name of the keyword
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Construct a keyword from a name.
    ///
    /// There are no safeguards in place for ensuring that the name given would be valid EDN.
    pub fn from_name(name: &str) -> Keyword {
        Keyword {
            namespace: Option::None,
            name: IStr::new(name),
        }
    }

    /// Construct a keyword from a namespace and a name.
    ///
    /// There are no safeguards in place for ensuring that either the namespace or the name
    /// given would be valid EDN.
    pub fn from_namespace_and_name(namespace: &str, name: &str) -> Keyword {
        Keyword {
            namespace: Option::Some(IStr::new(namespace)),
            name: IStr::new(name),
        }
    }
}

/// If the namespace and name of the symbol follow the proper rules, displaying
/// a keyword should give valid edn
impl Display for Keyword {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.namespace {
            Some(ns) => write!(f, ":{}/{}", ns, self.name),
            None => write!(f, ":{}", self.name),
        }
    }
}

/// A symbol, as described in EDN data model is an identifier.
///
/// Because its contents are interned, cloning and comparisons should be relatively
/// cheap operations, while construction is relatively not.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Symbol {
    namespace: Option<IStr>,
    name: IStr,
}

impl Symbol {
    /// The namespace of the symbol, if there is one
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_ref().map(|s| s.as_str())
    }

    /// The name of the symbol
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Construct a symbol from a name.
    ///
    /// There are no safeguards in place for ensuring that the name given would be valid EDN.
    pub fn from_name(name: &str) -> Symbol {
        Symbol {
            namespace: Option::None,
            name: IStr::new(name),
        }
    }

    /// Construct a symbol from a namespace and a name.
    ///
    /// There are no safeguards in place for ensuring that either the namespace or the name
    /// given would be valid EDN.
    pub fn from_namespace_and_name(namespace: &str, name: &str) -> Symbol {
        Symbol {
            namespace: Option::Some(IStr::new(namespace)),
            name: IStr::new(name),
        }
    }
}

/// If the namespace and name of the symbol follow the proper rules, displaying
/// a symbol should give valid edn
impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.namespace {
            Some(ns) => write!(f, "{}/{}", ns, self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

/// An Edn Value.
#[derive(Debug, Clone, Ord, PartialOrd, Eq)]
pub enum Value {
    Nil,
    String(String),
    Character(char),
    Symbol(Symbol),
    Keyword(Keyword),
    Integer(i64),
    Float(OrderedFloat<f64>),
    BigInt(BigInt),
    BigDec(BigDecimal),
    List(Vec<Value>),
    Vector(Vec<Value>),
    Map(BTreeMap<Value, Value>),
    Set(BTreeSet<Value>),
    Boolean(bool),
    Inst(chrono::DateTime<FixedOffset>),
    Uuid(Uuid),
    TaggedElement(Symbol, Box<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        equal(self, other)
    }
}

/// The errors that can be encountered during parsing.
#[derive(Debug, Error, PartialEq)]
pub enum ParserError {
    #[error("The input was entirely blank")]
    EmptyInput,

    #[error("Unexpected end of input")]
    UnexpectedEndOfInput,

    #[error("Invalid escape sequence in string")]
    InvalidStringEscape,

    #[error("Duplicate value in a set")]
    DuplicateValueInSet { value: Value },

    #[error("Duplicate key in a map")]
    DuplicateKeyInMap { value: Value },

    #[error("Only symbols, and optionally non-namespaced keywords, can be used as tags")]
    InvalidElementForTag { value: Value },

    #[error("Value is not a map")]
    NamespacedMapTagNeedsMap {
        namespace: String,
        got_instead_of_map: Value,
    },

    #[error("Invalid character specification")]
    InvalidCharacterSpecification,

    #[error("Unexpected character")]
    UnexpectedCharacter(char),

    #[error("Invalid keyword")]
    InvalidKeyword,

    #[error("Invalid Symbol")]
    InvalidSymbol(Symbol),

    #[error("Map must have an even number of elements")]
    OddNumberOfMapElements,

    #[error("Error parsing #inst")]
    InvalidInst(Option<chrono::format::ParseError>),

    #[error("Error parsing #uuid")]
    InvalidUuid(Option<uuid::Error>),

    #[error("Cannot have slash at the beginning of symbol")]
    CannotHaveSlashAtBeginningOfSymbol,

    #[error("Cannot have slash at the end of symbol")]
    CannotHaveSlashAtEndOfSymbol,

    #[error("Cannot have more than one slash in a symbol")]
    CannotHaveMoreThanOneSlashInSymbol,

    #[error("Cannot have slash at the beginning of symbol")]
    CannotHaveSlashAtBeginningOfKeyword,

    #[error("Cannot have slash at the end of symbol")]
    CannotHaveSlashAtEndOfKeyword,

    #[error("Cannot have more than one slash in a symbol")]
    CannotHaveMoreThanOneSlashInKeyword,

    #[error("Only 0 can start with 0")]
    OnlyZeroCanStartWithZero,

    #[error("Invalid float")]
    BadFloat {
        parsing: String,
        encountered: ParseFloatError,
    },

    #[error("Invalid int")]
    BadInt {
        parsing: String,
        encountered: ParseIntError,
    },

    #[error("Invalid big decimal")]
    BadBigDec {
        parsing: String,
        encountered: ParseBigDecimalError,
    },

    #[error("Invalid big int")]
    BadBigInt {
        parsing: String,
        encountered: ParseBigIntError,
    },

    #[error("Unexpected Extra Input")]
    ExtraInput {
        parsed_value: Value,
        extra_input: Vec<char>,
    },

    #[error("Error with context")]
    WithContext {
        context: Vec<Context>,
        row_col: RowCol,
        error: Box<ParserError>,
    },
}

struct ParserSuccess<'a> {
    remaining_input: &'a [char],
    value: Value,
}

#[derive(Debug)]
enum ParserState {
    Begin,
    ParsingList {
        values_so_far: Vec<Value>,
    },
    ParsingVector {
        values_so_far: Vec<Value>,
    },
    ParsingMap {
        values_so_far: Vec<Value>,
    },
    ParsingSet {
        values_so_far: Vec<Value>,
    },
    ParsingSymbol {
        characters_before_a_slash: Vec<char>,
        characters_after_a_slash: Vec<char>,
        saw_slash: bool,
    }, // Decide after parsing symbol if it is true, false, nil, or actually supposed to be a number
    ParsingString {
        built_up: String,
    },
    ParsingCharacter,
    SelectingDispatch,
}

/// Commas are considered whitespace for EDN
fn is_whitespace(c: char) -> bool {
    c.is_whitespace() || c == ','
}

fn is_allowed_symbol_character(c: char) -> bool {
    c == '.'
        || c == '*'
        || c == '+'
        || c == '!'
        || c == '-'
        || c == '_'
        || c == '?'
        || c == '$'
        || c == '%'
        || c == '&'
        || c == '='
        || c == '<'
        || c == '>'
        || c.is_alphabetic()
        // Technically this is *not true*, but the plan is to parse all
        // numbers as symbols and figure out later which should be numbers
        // and if any are malformed there i will produce an error
        || c.is_numeric()
}

fn equal(v1: &Value, v2: &Value) -> bool {
    match (v1, v2) {
        // nil, booleans, strings, characters, and symbols
        // are equal to values of the same type with the same edn representation
        (Value::Nil, Value::Nil) => true,
        (Value::Boolean(b1), Value::Boolean(b2)) => b1 == b2,
        (Value::String(s1), Value::String(s2)) => s1 == s2,
        (Value::Character(c1), Value::Character(c2)) => c1 == c2,
        (Value::Symbol(s1), Value::Symbol(s2)) => s1 == s2,
        (Value::Keyword(k1), Value::Keyword(k2)) => k1 == k2,

        // integers and floating point numbers should be considered equal to values only of the
        // same magnitude, type, and precision. Comingling numeric types and precision in
        // map/set key/elements, or constituents therein, is not advised.
        (Value::Float(f1), Value::Float(f2)) => f1 == f2,
        (Value::Integer(i1), Value::Integer(i2)) => i1 == i2,
        (Value::BigInt(bi1), Value::BigInt(bi2)) => bi1 == bi2,
        (Value::BigDec(bd1), Value::BigDec(bd2)) => bd1 == bd2,

        // sequences (lists and vectors) are equal to other sequences whose count
        // of elements is the same, and for which each corresponding pair of
        // elements (by ordinal) is equal.
        (Value::List(vals1) | Value::Vector(vals1), Value::List(vals2) | Value::Vector(vals2)) => {
            if vals1.len() != vals2.len() {
                false
            } else {
                vals1
                    .iter()
                    .zip(vals2.iter())
                    .fold(true, |all_same, (v1, v2)| all_same && equal(v1, v2))
            }
        }

        // sets are equal if they have the same count of elements and,
        // for every element in one set, an equal element is in the other.
        (Value::Set(vals1), Value::Set(vals2)) => vals1 == vals2,

        // maps are equal if they have the same number of entries,
        // and for every key/value entry in one map an equal key is present
        // and mapped to an equal value in the other.
        (Value::Map(entries1), Value::Map(entries2)) => entries1 == entries2,

        // tagged elements must define their own equality semantics.
        // #uuid elements are equal if their canonic representations are equal.
        // #inst elements are equal if their representation strings designate
        // the same timestamp per RFC-3339.
        (Value::Uuid(uuid1), Value::Uuid(uuid2)) => uuid1 == uuid2,
        (Value::Inst(dt1), Value::Inst(dt2)) => dt1 == dt2,
        (Value::TaggedElement(tag1, value1), Value::TaggedElement(tag2, value2)) => {
            equal(&Value::Symbol(tag1.clone()), &Value::Symbol(tag2.clone()))
                && equal(value1, value2)
        }

        _ => false,
    }
}

/// Struct holding a row position and a column position.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct RowCol {
    /// The row. 1 indexed. 0 if no characters on the line have been read.
    pub row: usize,
    /// The col. 1 indexed. Starts at 1.
    pub col: usize,
}

/// The purpose of this interface is to keep track of a "context stack"
/// so i can give better errors if parsing fails. This only has events for
/// things that might have errors "within" some structure
trait ParseObserver {
    fn start_parsing_vector(&mut self);

    fn start_parsing_list(&mut self);

    fn start_parsing_map(&mut self);

    fn start_parsing_set(&mut self);

    fn start_parsing_string(&mut self);

    fn stop_parsing_current(&mut self);

    fn advance_one_char_from(&mut self, start: &[char]) {
        self.advance_n_chars_from(start, 1);
    }

    fn advance_n_chars_from(&mut self, start: &[char], n: usize);
}

/// No op observer, for when we want to call the parser recursively consequence free.
struct NoOpParseObserver;

impl ParseObserver for NoOpParseObserver {
    fn start_parsing_vector(&mut self) {}

    fn start_parsing_list(&mut self) {}

    fn start_parsing_map(&mut self) {}

    fn start_parsing_set(&mut self) {}

    fn start_parsing_string(&mut self) {}

    fn stop_parsing_current(&mut self) {}

    fn advance_n_chars_from(&mut self, _start: &[char], _n: usize) {}
}

/// An element of context about what the parser was doing when an error was detected.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Context {
    ParsingVector(RowCol),
    ParsingList(RowCol),
    ParsingMap(RowCol),
    ParsingSet(RowCol),
    ParsingString(RowCol),
}

#[derive(Debug)]
struct ContextStackerObserver {
    context: Vec<Context>,
    row_col: RowCol,
}

impl ContextStackerObserver {
    fn new() -> ContextStackerObserver {
        ContextStackerObserver {
            context: Vec::new(),
            row_col: RowCol { row: 0, col: 1 },
        }
    }
}

impl ParseObserver for ContextStackerObserver {
    fn start_parsing_vector(&mut self) {
        self.context.push(Context::ParsingVector(self.row_col));
    }

    fn start_parsing_list(&mut self) {
        self.context.push(Context::ParsingList(self.row_col));
    }

    fn start_parsing_map(&mut self) {
        self.context.push(Context::ParsingMap(self.row_col));
    }

    fn start_parsing_set(&mut self) {
        self.context.push(Context::ParsingSet(self.row_col));
    }

    fn start_parsing_string(&mut self) {
        self.context.push(Context::ParsingString(self.row_col));
    }

    fn stop_parsing_current(&mut self) {
        self.context.pop();
    }

    fn advance_n_chars_from(&mut self, start: &[char], n: usize) {
        let mut i = 0;
        while i < start.len() && i < n {
            if start[i] == '\n' {
                self.row_col.row = 0;
                self.row_col.col += 1;
            } else {
                self.row_col.row += 1;
            }
            i += 1;
        }
    }
}
/// Likely suboptimal parsing. Focus for now is just on getting correct results.
fn parse_helper<'a, Observer: ParseObserver>(
    mut s: &'a [char],
    mut parser_state: ParserState,
    observer: &mut Observer,
    opts: &ParserOptions,
) -> Result<ParserSuccess<'a>, ParserError> {
    'parsing: loop {
        // Strip out comments
        match parser_state {
            ParserState::ParsingString { .. } => {}
            _ => {
                if !s.is_empty() && s[0] == ';' {
                    while !s.is_empty() && s[0] != '\n' {
                        observer.advance_one_char_from(s);
                        s = &s[1..];
                    }
                }
            }
        };
        match &mut parser_state {
            ParserState::Begin => {
                if s.is_empty() {
                    return Err(ParserError::EmptyInput);
                } else if is_whitespace(s[0]) {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                    parser_state = ParserState::Begin;
                } else if s[0] == '(' {
                    observer.advance_one_char_from(s);
                    observer.start_parsing_list();
                    s = &s[1..];
                    parser_state = ParserState::ParsingList {
                        values_so_far: vec![],
                    };
                } else if s[0] == '[' {
                    observer.advance_one_char_from(s);
                    observer.start_parsing_vector();
                    s = &s[1..];
                    parser_state = ParserState::ParsingVector {
                        values_so_far: vec![],
                    };
                } else if s[0] == '{' {
                    observer.advance_one_char_from(s);
                    observer.start_parsing_map();
                    s = &s[1..];
                    parser_state = ParserState::ParsingMap {
                        values_so_far: vec![],
                    };
                } else if s[0] == '"' {
                    observer.advance_one_char_from(s);
                    observer.start_parsing_string();
                    s = &s[1..];
                    parser_state = ParserState::ParsingString {
                        built_up: "".to_string(),
                    };
                } else if s[0] == ':' {
                    // For parsing a keyword, we can just fall back on the logic for parsing a symbol
                    // **somewhat** of a hack and it means we need to move the logic for converting
                    // symbols for true, false, and nil higher up and the error messages might
                    // end up strange but i am okay with that.
                    observer.advance_one_char_from(s);
                    let ParserSuccess {
                        remaining_input,
                        value,
                    } = parse_helper(&s[1..], ParserState::Begin, observer, opts).map_err(
                        |err| match err {
                            ParserError::CannotHaveSlashAtBeginningOfSymbol => {
                                ParserError::CannotHaveSlashAtBeginningOfKeyword
                            }
                            ParserError::CannotHaveSlashAtEndOfSymbol => {
                                ParserError::CannotHaveSlashAtEndOfKeyword
                            }
                            ParserError::CannotHaveMoreThanOneSlashInSymbol => {
                                ParserError::CannotHaveMoreThanOneSlashInKeyword
                            }
                            ParserError::EmptyInput => ParserError::InvalidKeyword,
                            err => err,
                        },
                    )?;
                    if let Value::Symbol(symbol) = value {
                        return Ok(ParserSuccess {
                            remaining_input,
                            value: Value::Keyword(Keyword {
                                namespace: symbol.namespace,
                                name: symbol.name,
                            }),
                        });
                    } else {
                        return Err(ParserError::InvalidKeyword);
                    }
                } else if s[0] == '\\' {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                    parser_state = ParserState::ParsingCharacter;
                } else if s[0] == '#' {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                    parser_state = ParserState::SelectingDispatch;
                } else if is_allowed_symbol_character(s[0]) || s[0] == '/' {
                    parser_state = ParserState::ParsingSymbol {
                        characters_before_a_slash: vec![],
                        characters_after_a_slash: vec![],
                        saw_slash: false,
                    };
                } else {
                    return Err(ParserError::UnexpectedCharacter(s[0]));
                }
            }

            ParserState::ParsingList {
                ref mut values_so_far,
            } => {
                if s.is_empty() {
                    return Err(ParserError::UnexpectedEndOfInput);
                } else if is_whitespace(s[0]) {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                } else if s[0] == ')' {
                    observer.stop_parsing_current();
                    observer.advance_one_char_from(s);
                    return Ok(ParserSuccess {
                        remaining_input: &s[1..],
                        value: Value::List(values_so_far.clone()),
                    });
                } else {
                    let ParserSuccess {
                        remaining_input,
                        value,
                    } = parse_helper(s, ParserState::Begin, observer, opts)?;
                    values_so_far.push(value);
                    s = remaining_input;
                }
            }

            // Almost total duplicate of ParsingList
            ParserState::ParsingVector {
                ref mut values_so_far,
            } => {
                if s.is_empty() {
                    return Err(ParserError::UnexpectedEndOfInput);
                } else if is_whitespace(s[0]) {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                } else if s[0] == ']' {
                    observer.stop_parsing_current();
                    observer.advance_one_char_from(s);
                    return Ok(ParserSuccess {
                        remaining_input: &s[1..],
                        value: Value::Vector(values_so_far.clone()),
                    });
                } else {
                    let ParserSuccess {
                        remaining_input,
                        value,
                    } = parse_helper(s, ParserState::Begin, observer, opts)?;
                    values_so_far.push(value);
                    s = remaining_input;
                }
            }

            ParserState::ParsingMap {
                ref mut values_so_far,
            } => {
                if s.is_empty() {
                    return Err(ParserError::UnexpectedEndOfInput);
                } else if is_whitespace(s[0]) {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                } else if s[0] == '}' {
                    if values_so_far.len() % 2 != 0 {
                        return Err(ParserError::OddNumberOfMapElements);
                    } else {
                        // I'm confident there has to be a better way to do this
                        let entries: Vec<(Value, Value)> = values_so_far
                            .iter_mut()
                            .map(|v| v.clone())
                            .batching(|it| match it.next() {
                                None => None,
                                Some(x) => it.next().map(|y| (x, y)),
                            })
                            .collect();

                        let mut seen = BTreeSet::new();
                        for (k, _) in entries.iter() {
                            if seen.contains(k) {
                                return Err(ParserError::DuplicateKeyInMap { value: k.clone() });
                            }
                            seen.insert(k);
                        }
                        let value = Value::Map(BTreeMap::from_iter(entries));

                        observer.stop_parsing_current();
                        observer.advance_one_char_from(s);
                        return Ok(ParserSuccess {
                            remaining_input: &s[1..],
                            value,
                        });
                    }
                } else {
                    let ParserSuccess {
                        remaining_input,
                        value,
                    } = parse_helper(s, ParserState::Begin, observer, opts)?;
                    values_so_far.push(value);
                    s = remaining_input;
                }
            }

            ParserState::ParsingSet {
                ref mut values_so_far,
            } => {
                if s.is_empty() {
                    return Err(ParserError::UnexpectedEndOfInput);
                } else if is_whitespace(s[0]) {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                } else if s[0] == '}' {
                    let mut seen = BTreeSet::new();
                    for v in values_so_far.iter() {
                        if seen.contains(v) {
                            return Err(ParserError::DuplicateValueInSet { value: v.clone() });
                        }
                        seen.insert(v);
                    }
                    observer.stop_parsing_current();
                    observer.advance_one_char_from(s);
                    return Ok(ParserSuccess {
                        remaining_input: &s[1..],
                        value: Value::Set(values_so_far.iter().cloned().collect()),
                    });
                } else {
                    let ParserSuccess {
                        remaining_input,
                        value,
                    } = parse_helper(s, ParserState::Begin, observer, opts)?;
                    values_so_far.push(value);
                    s = remaining_input;
                }
            }

            ParserState::ParsingSymbol {
                ref mut characters_before_a_slash,
                ref mut characters_after_a_slash,
                ref mut saw_slash,
            } => {
                if s.is_empty() {
                    if characters_before_a_slash.is_empty() {
                        return Err(ParserError::UnexpectedEndOfInput);
                    } else if characters_after_a_slash.is_empty() {
                        if *saw_slash {
                            return Err(ParserError::UnexpectedEndOfInput);
                        } else {
                            let name: String =
                                characters_before_a_slash.iter_mut().map(|c| *c).collect();
                            return Ok(ParserSuccess {
                                remaining_input: s,
                                value: Value::Symbol(Symbol::from_name(&name)),
                            });
                        }
                    } else {
                        let namespace: String =
                            characters_before_a_slash.iter_mut().map(|c| *c).collect();
                        let name: String =
                            characters_after_a_slash.iter_mut().map(|c| *c).collect();
                        return Ok(ParserSuccess {
                            remaining_input: s,
                            value: Value::Symbol(Symbol::from_namespace_and_name(
                                &namespace, &name,
                            )),
                        });
                    }
                } else if characters_before_a_slash.is_empty() && !*saw_slash {
                    if is_allowed_symbol_character(s[0]) {
                        characters_before_a_slash.push(s[0]);
                        observer.advance_one_char_from(s);
                        s = &s[1..];
                    } else if s[0] == '/' {
                        if s.len() > 1 && is_allowed_symbol_character(s[1]) {
                            return Err(ParserError::CannotHaveSlashAtBeginningOfSymbol);
                        } else {
                            observer.advance_one_char_from(s);
                            return Ok(ParserSuccess {
                                remaining_input: &s[1..],
                                value: Value::Symbol(Symbol::from_name("/")),
                            });
                        }
                    } else {
                        return Err(ParserError::UnexpectedCharacter(s[0]));
                    }
                } else if !*saw_slash {
                    if is_allowed_symbol_character(s[0]) {
                        characters_before_a_slash.push(s[0]);
                        observer.advance_one_char_from(s);
                        s = &s[1..];
                    } else if s[0] == '/' {
                        if s.len() == 1 || (s.len() > 1 && !is_allowed_symbol_character(s[1])) {
                            return Err(ParserError::CannotHaveSlashAtEndOfSymbol);
                        } else {
                            observer.advance_one_char_from(s);
                            s = &s[1..];
                            *saw_slash = true;
                        }
                    } else {
                        let name: String =
                            characters_before_a_slash.iter_mut().map(|c| *c).collect();
                        return Ok(ParserSuccess {
                            remaining_input: s,
                            value: Value::Symbol(Symbol::from_name(&name)),
                        });
                    }
                } else if is_allowed_symbol_character(s[0]) {
                    characters_after_a_slash.push(s[0]);
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                } else if s[0] == '/' {
                    return Err(ParserError::CannotHaveMoreThanOneSlashInSymbol);
                } else {
                    let namespace: String =
                        characters_before_a_slash.iter_mut().map(|c| *c).collect();
                    let name: String = characters_after_a_slash.iter_mut().map(|c| *c).collect();
                    return Ok(ParserSuccess {
                        remaining_input: s,
                        value: Value::Symbol(Symbol::from_namespace_and_name(&namespace, &name)),
                    });
                }
            }

            ParserState::ParsingString { ref mut built_up } => {
                if s.is_empty() {
                    return Err(ParserError::UnexpectedEndOfInput);
                } else if s[0] == '"' {
                    observer.stop_parsing_current();
                    observer.advance_one_char_from(s);
                    return Ok(ParserSuccess {
                        remaining_input: &s[1..],
                        value: Value::String(built_up.clone()),
                    });
                } else if s[0] == '\\' {
                    if s.is_empty() {
                        return Err(ParserError::InvalidStringEscape);
                    } else {
                        match s[1] {
                            't' => built_up.push('\t'),
                            'r' => built_up.push('\r'),
                            'n' => built_up.push('\n'),

                            '\\' => built_up.push('\\'),
                            '"' => built_up.push('"'),
                            'u' => {
                                if s.len() >= 5 {
                                    let str: String = s[2..6].iter().copied().collect();
                                    let unicode = u32::from_str_radix(&str, 16)
                                        .map_err(|_| ParserError::InvalidStringEscape)?;
                                    match char::from_u32(unicode) {
                                        None => return Err(ParserError::InvalidStringEscape),
                                        Some(c) => built_up.push(c),
                                    }
                                    observer.advance_n_chars_from(s, 5);
                                    s = &s[6..];
                                    continue 'parsing;
                                } else {
                                    return Err(ParserError::InvalidStringEscape);
                                }
                            }
                            _ => return Err(ParserError::InvalidStringEscape),
                        }
                        observer.advance_n_chars_from(s, 2);
                        s = &s[2..];
                    }
                } else {
                    built_up.push(s[0]);
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                }
            }
            ParserState::SelectingDispatch => {
                if s.is_empty() {
                    return Err(ParserError::UnexpectedEndOfInput);
                } else if s[0] == '_' {
                    // Drop the next form. Still error if that form is malformed
                    observer.advance_one_char_from(s);
                    let ParserSuccess {
                        remaining_input, ..
                    } = parse_helper(&s[1..], ParserState::Begin, observer, opts)?;
                    s = remaining_input;
                } else if s[0] == '{' {
                    observer.advance_one_char_from(s);
                    s = &s[1..];
                    observer.start_parsing_set();
                    parser_state = ParserState::ParsingSet {
                        values_so_far: vec![],
                    };
                } else {
                    // We expect to read a symbol next and we will associate that symbol as the tag of
                    // the following element
                    let ParserSuccess {
                        remaining_input,
                        value,
                    } = parse_helper(s, ParserState::Begin, observer, opts)?;
                    match value {
                        Value::Symbol(symbol) => {
                            let next_success =
                                parse_helper(remaining_input, ParserState::Begin, observer, opts)?;

                            // Handle builtin #inst
                            if symbol.namespace == None && symbol.name == "inst" {
                                if let Value::String(timestamp) = next_success.value {
                                    let datetime = chrono::DateTime::parse_from_rfc3339(&timestamp)
                                        .map_err(|parse_error| {
                                            ParserError::InvalidInst(Some(parse_error))
                                        })?;
                                    return Ok(ParserSuccess {
                                        remaining_input: next_success.remaining_input,
                                        value: Value::Inst(datetime),
                                    });
                                } else {
                                    return Err(ParserError::InvalidInst(None));
                                }
                            }
                            // Handle builtin #uuid
                            else if symbol.namespace == None && symbol.name == "uuid" {
                                if let Value::String(uuid_str) = next_success.value {
                                    let uuid =
                                        Uuid::parse_str(&uuid_str).map_err(|parse_error| {
                                            ParserError::InvalidUuid(Some(parse_error))
                                        })?;
                                    return Ok(ParserSuccess {
                                        remaining_input: next_success.remaining_input,
                                        value: Value::Uuid(uuid),
                                    });
                                } else {
                                    return Err(ParserError::InvalidUuid(None));
                                }
                            }
                            // Everything else becomes a generic TaggedElement
                            else {
                                return Ok(ParserSuccess {
                                    remaining_input: next_success.remaining_input,
                                    value: Value::TaggedElement(
                                        symbol,
                                        Box::new(next_success.value),
                                    ),
                                });
                            }
                        }
                        Value::Keyword(ref ns) => {
                            if !opts.allow_namespaced_map_syntax || ns.namespace.is_some() {
                                return Err(ParserError::InvalidElementForTag { value });
                            } else {
                                let next_success = parse_helper(
                                    remaining_input,
                                    ParserState::Begin,
                                    observer,
                                    opts,
                                )?;
                                if let Value::Map(following_map) = next_success.value {
                                    let mut new_map = BTreeMap::new();
                                    for (k, v) in following_map.into_iter() {
                                        new_map.insert(
                                            match &k {
                                                Value::Keyword(keyword) => {
                                                    if keyword.namespace.is_some() {
                                                        k
                                                    } else {
                                                        Value::Keyword(
                                                            Keyword::from_namespace_and_name(
                                                                &ns.name,
                                                                &keyword.name,
                                                            ),
                                                        )
                                                    }
                                                }
                                                _ => k,
                                            },
                                            v,
                                        );
                                    }
                                    return Ok(ParserSuccess {
                                        remaining_input: next_success.remaining_input,
                                        value: Value::Map(new_map),
                                    });
                                } else {
                                    return Err(ParserError::NamespacedMapTagNeedsMap {
                                        namespace: ns.name.to_string(),
                                        got_instead_of_map: value.clone(),
                                    });
                                }
                            }
                        }
                        _ => return Err(ParserError::InvalidElementForTag { value }),
                    }
                }
            }

            ParserState::ParsingCharacter => {
                let ParserSuccess {
                    remaining_input,
                    value,
                } = parse_helper(s, ParserState::Begin, observer, opts)?;

                if let Value::Symbol(symbol) = value {
                    if symbol.namespace == None {
                        if symbol.name.len() == 1 {
                            return Ok(ParserSuccess {
                                remaining_input,
                                value: Value::Character(symbol.name.chars().next().expect(
                                    "Asserted that this string has at least one character.",
                                )),
                            });
                        } else {
                            return match symbol.name.as_str() {
                                "newline" => Ok(ParserSuccess {
                                    remaining_input,
                                    value: Value::Character('\n'),
                                }),
                                "return" => Ok(ParserSuccess {
                                    remaining_input,
                                    value: Value::Character('\r'),
                                }),
                                "space" => Ok(ParserSuccess {
                                    remaining_input,
                                    value: Value::Character(' '),
                                }),
                                "tab" => Ok(ParserSuccess {
                                    remaining_input,
                                    value: Value::Character('\t'),
                                }),
                                _ => Err(ParserError::InvalidCharacterSpecification),
                            };
                        }
                    } else {
                        return Err(ParserError::InvalidCharacterSpecification);
                    }
                } else {
                    return Err(ParserError::InvalidCharacterSpecification);
                }
            }
        }
    }
}

/// Crawls the tree mutably to avoid allocations
fn replace_nil_false_true(value: &mut Value) {
    match value {
        Value::Symbol(symbol) => {
            if symbol.namespace == None {
                if symbol.name == "true" {
                    *value = Value::Boolean(true)
                } else if symbol.name == "false" {
                    *value = Value::Boolean(false)
                } else if symbol.name == "nil" {
                    *value = Value::Nil
                }
            }
        }
        Value::List(elements) => {
            for element in elements.iter_mut() {
                replace_nil_false_true(element);
            }
        }
        Value::Vector(elements) => {
            for element in elements.iter_mut() {
                replace_nil_false_true(element);
            }
        }
        Value::Map(entries) => {
            let mut new_map = BTreeMap::new();
            for (k, v) in entries.iter_mut() {
                let mut k2 = k.clone();
                replace_nil_false_true(&mut k2);
                replace_nil_false_true(v);
                new_map.insert(k2, v.clone());
            }
            *value = Value::Map(new_map)
        }
        Value::Set(elements) => {
            let mut new_set = BTreeSet::new();
            for element in elements.iter() {
                let mut element = element.clone();
                replace_nil_false_true(&mut element);
                new_set.insert(element);
            }
            *value = Value::Set(new_set);
        }
        Value::TaggedElement(_, val) => replace_nil_false_true(val),
        _ => {}
    }
}

/// Previous parsing step interprets all numbers as symbols. This step
/// goes through all the symbols and re-interprets them as numbers
/// as appropriate.
///
/// This is one are for potential improvement, as doing it this way means
/// that I can't correctly tie these errors to the lines they came from.
fn replace_numeric_types(value: &mut Value) -> Result<(), ParserError> {
    let starts_bad = |name: &str| {
        name.starts_with(|c: char| c.is_numeric())
            || (name.len() > 1 && name[1..].starts_with(|c: char| c.is_numeric()))
    };
    match value {
        Value::Symbol(symbol) => {
            match &symbol.namespace {
                Some(ns) => {
                    if starts_bad(ns) || starts_bad(&symbol.name) {
                        return Err(ParserError::InvalidSymbol(symbol.clone()));
                    }
                }
                None => {
                    // See if it starts wrong
                    // Symbols begin with a non-numeric character
                    // If -, + or . are the first character, the second character (if any) must be non-numeric.
                    let name: &str = &symbol.name;
                    if starts_bad(name) {
                        if name.ends_with('M') && name.chars().filter(|c| *c == 'M').count() == 1 {
                            *value = Value::BigDec(
                                str::parse::<BigDecimal>(&symbol.name[..symbol.name.len() - 1])
                                    .map_err(|err| ParserError::BadBigDec {
                                        parsing: symbol.name.to_string(),
                                        encountered: err,
                                    })?,
                            );
                        } else if name.contains('.') || name.contains('e') || name.contains('E') {
                            *value = Value::Float(OrderedFloat(
                                str::parse::<f64>(&symbol.name).map_err(|err| {
                                    ParserError::BadFloat {
                                        parsing: symbol.name.to_string(),
                                        encountered: err,
                                    }
                                })?,
                            ));
                        } else if name != "0" && (name.starts_with('0'))
                            || (name != "+0"
                                && (name.len() > 1
                                    && name.starts_with('+')
                                    && name[1..].starts_with('0')))
                            || (name != "-0"
                                && (name.len() > 1
                                    && name.starts_with('-')
                                    && name[1..].starts_with('0')))
                        {
                            // Only ints are subject to this restriction it seems
                            return Err(ParserError::OnlyZeroCanStartWithZero);
                        } else if name.ends_with('N')
                            && name.chars().filter(|c| *c == 'N').count() == 1
                        {
                            *value = Value::BigInt(
                                str::parse::<BigInt>(&symbol.name[..symbol.name.len() - 1])
                                    .map_err(|err| ParserError::BadBigInt {
                                        parsing: symbol.name.to_string(),
                                        encountered: err,
                                    })?,
                            );
                        } else {
                            *value =
                                Value::Integer(str::parse::<i64>(&symbol.name).map_err(|err| {
                                    ParserError::BadInt {
                                        parsing: symbol.name.to_string(),
                                        encountered: err,
                                    }
                                })?);
                        }
                    }
                }
            }
            Ok(())
        }
        Value::List(elements) => {
            for element in elements {
                replace_numeric_types(element)?;
            }
            Ok(())
        }
        Value::Vector(elements) => {
            for element in elements {
                replace_numeric_types(element)?;
            }
            Ok(())
        }
        Value::Map(entries) => {
            let mut new_map = BTreeMap::new();
            for (k, v) in entries.iter_mut() {
                let mut k2 = k.clone();
                replace_numeric_types(&mut k2)?;
                replace_numeric_types(v)?;
                new_map.insert(k2, v.clone());
            }
            *value = Value::Map(new_map);
            Ok(())
        }
        Value::Set(elements) => {
            let mut new_set = BTreeSet::new();
            for element in elements.iter() {
                let mut new_element = element.clone();
                replace_numeric_types(&mut new_element)?;
                new_set.insert(new_element);
            }
            *value = Value::Set(new_set);
            Ok(())
        }
        Value::TaggedElement(_, val) => replace_numeric_types(val),
        _ => Ok(()),
    }
}

/// Options you can pass to the EDN parser
#[derive(Debug, Copy, Clone)]
pub struct ParserOptions {
    /// Whether to return errors with the context of what forms the parser was
    /// parsing and the line numbers where the errors occurred.
    ///
    /// This is currently done on a "best effort" basis and some errors, like those that arise
    /// parsing numbers, are currently not eligible for this tracking
    ///
    /// Defaults to false.
    pub track_line_numbers: bool,
    /// Whether or not to error if there exists non-whitespace content after
    /// a valid EDN form.
    ///
    /// Defaults to false.
    pub allow_extra_input: bool,
    /// Whether to allow the #some.ns{:key "val"} syntax that was introduced in clojure 1.9
    /// but not reflected in the EDN spec.
    ///
    /// Defaults to true.
    pub allow_namespaced_map_syntax: bool,
}

impl Default for ParserOptions {
    fn default() -> Self {
        ParserOptions {
            track_line_numbers: false,
            allow_extra_input: false,
            allow_namespaced_map_syntax: true,
        }
    }
}

/// Parse EDN from the given input string with the given options.
pub fn parse_with_options(s: &str, opts: ParserOptions) -> Result<Value, ParserError> {
    let chars: Vec<char> = s.chars().collect();
    if opts.track_line_numbers {
        let mut context = ContextStackerObserver::new();
        let ParserSuccess {
            remaining_input,
            mut value,
        } = parse_helper(&chars, ParserState::Begin, &mut context, &opts).map_err(|err| {
            ParserError::WithContext {
                context: context.context.clone(),
                row_col: context.row_col,
                error: Box::new(err),
            }
        })?;
        if !opts.allow_extra_input {
            for c in remaining_input {
                if !is_whitespace(*c) {
                    return Err(ParserError::WithContext {
                        context: context.context.clone(),
                        row_col: context.row_col,
                        error: Box::new(ParserError::ExtraInput {
                            parsed_value: value,
                            extra_input: remaining_input.to_vec(),
                        }),
                    });
                }
            }
        }
        replace_nil_false_true(&mut value);
        replace_numeric_types(&mut value)?;
        Ok(value)
    } else {
        let mut context = NoOpParseObserver;
        let ParserSuccess {
            remaining_input,
            mut value,
        } = parse_helper(&chars, ParserState::Begin, &mut context, &opts)?;
        if !opts.allow_extra_input {
            for c in remaining_input {
                if !is_whitespace(*c) {
                    return Err(ParserError::ExtraInput {
                        parsed_value: value,
                        extra_input: remaining_input.to_vec(),
                    });
                };
            }
        }

        replace_nil_false_true(&mut value);
        replace_numeric_types(&mut value)?;
        Ok(value)
    }
}

/// Parse Edn from the given input string with default options.
pub fn parse(s: &str) -> Result<Value, ParserError> {
    parse_with_options(s, ParserOptions::default())
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => {
                write!(f, "nil")?;
            }
            Value::String(s) => {
                write!(f, "\"")?;
                for c in s.chars() {
                    match c {
                        '\t' => write!(f, "\\t")?,
                        '\r' => write!(f, "\\r")?,
                        '\n' => write!(f, "\\n")?,

                        '\\' => write!(f, "\\\\")?,
                        '\"' => write!(f, "\\\"")?,
                        _ => write!(f, "{}", c)?,
                    };
                }
                write!(f, "\"")?;
            }
            Value::Character(c) => {
                match c {
                    '\n' => write!(f, "\\newline")?,
                    '\r' => write!(f, "\\return")?,
                    ' ' => write!(f, "\\space")?,
                    '\t' => write!(f, "\\tab")?,
                    _ => write!(f, "{}", c)?,
                };
            }
            Value::Symbol(symbol) => {
                write!(f, "{}", symbol)?;
            }
            Value::Keyword(keyword) => {
                write!(f, "{}", keyword)?;
            }
            Value::Integer(i) => {
                write!(f, "{}", i)?;
            }
            Value::Float(fl) => {
                write!(f, "{}", fl)?;
            }
            Value::BigInt(bi) => {
                write!(f, "{}N", bi)?;
            }
            Value::BigDec(bd) => {
                write!(f, "{}M", bd)?;
            }
            Value::List(elements) => {
                write!(f, "(")?;
                let mut i = 0;
                while i < elements.len() {
                    write!(f, "{}", elements[i])?;
                    if i != elements.len() - 1 {
                        write!(f, " ")?;
                    }
                    i += 1;
                }
                write!(f, ")")?;
            }
            Value::Vector(elements) => {
                write!(f, "[")?;
                let mut i = 0;
                while i < elements.len() {
                    write!(f, "{}", elements[i])?;
                    if i != elements.len() - 1 {
                        write!(f, " ")?;
                    }
                    i += 1;
                }
                write!(f, "]")?;
            }
            Value::Map(entries) => {
                write!(f, "{{")?;
                let mut i = 0;
                let entries: Vec<_> = entries.iter().collect();
                while i < entries.len() {
                    let (k, v) = &entries[i];
                    write!(f, "{} {}", k, v)?;
                    if i != entries.len() - 1 {
                        write!(f, " ")?;
                    }
                    i += 1;
                }
                write!(f, "}}")?;
            }
            Value::Set(elements) => {
                write!(f, "#{{")?;
                let elements: Vec<_> = elements.iter().collect();
                let mut i = 0;
                while i < elements.len() {
                    write!(f, "{}", elements[i])?;
                    if i != elements.len() - 1 {
                        write!(f, " ")?;
                    }
                    i += 1;
                }
                write!(f, "}}")?;
            }
            Value::Boolean(b) => {
                write!(f, "{}", b)?;
            }
            Value::Inst(inst) => {
                write!(f, "#inst \"{}\"", inst.to_rfc3339())?;
            }
            Value::Uuid(uuid) => {
                write!(f, "#uuid \"{}\"", uuid)?;
            }
            Value::TaggedElement(tag, value) => {
                write!(f, "#{} {}", tag, value)?;
            }
        }
        Ok(())
    }
}

/// Emit the given Edn value as a String.
pub fn emit(value: &Value) -> String {
    format!("{}", value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::DateTime;
    use std::iter::FromIterator;
    use std::vec;

    #[test]
    fn test_display_symbol() {
        assert_eq!(format!("{}", Symbol::from_name("abc")), "abc");
        assert_eq!(
            format!("{}", Symbol::from_namespace_and_name("abc", "def")),
            "abc/def"
        );
    }

    #[test]
    fn test_display_keyword() {
        assert_eq!(format!("{}", Keyword::from_name("abc")), ":abc");
        assert_eq!(
            format!("{}", Keyword::from_namespace_and_name("abc", "def")),
            ":abc/def"
        );
    }

    #[test]
    fn test_parsing_empty_list() {
        assert_eq!(Value::List(vec![]), parse("()").unwrap())
    }

    #[test]
    fn test_parsing_empty_vector() {
        assert_eq!(Value::Vector(vec![]), parse("[]").unwrap())
    }

    #[test]
    fn test_parsing_nested_empty_collections() {
        assert_eq!(
            Value::Vector(vec![
                Value::List(vec![]),
                Value::Vector(vec![]),
                Value::List(vec![]),
                Value::Vector(vec![])
            ]),
            parse("[()[]()[]]").unwrap()
        )
    }

    #[test]
    fn test_parsing_nested_empty_collections_with_whitespace() {
        assert_eq!(
            Value::Vector(vec![
                Value::List(vec![]),
                Value::Vector(vec![]),
                Value::List(vec![]),
                Value::Vector(vec![])
            ]),
            parse("   ,, , [ ,, , ,()[,,,]( ) []]").unwrap()
        )
    }

    #[test]
    fn test_parsing_empty_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![])),
            parse("{}").unwrap()
        )
    }

    #[test]
    fn test_parsing_uneven_map() {
        assert_eq!(Err(ParserError::OddNumberOfMapElements), parse("{()}"));
        assert_eq!(
            Err(ParserError::OddNumberOfMapElements),
            parse("{() [] []}")
        )
    }

    #[test]
    fn test_parsing_even_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::List(vec![]),
                Value::List(vec![])
            )])),
            parse("{() ()}").unwrap()
        );
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![
                (Value::List(vec![]), Value::Vector(vec![])),
                (Value::Keyword(Keyword::from_name("a")), Value::List(vec![]))
            ])),
            parse("{()[] :a ()}").unwrap()
        )
    }

    #[test]
    fn test_parsing_duplicate_map_keys() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::List(vec![]),
                Value::List(vec![])
            )])),
            parse("{() ()}").unwrap()
        );
        assert_eq!(
            Err(ParserError::DuplicateKeyInMap {
                value: Value::List(vec![])
            }),
            parse("{()[] () ()}")
        )
    }

    #[test]
    fn test_equals_for_list_and_vector() {
        assert!(equal(&Value::List(vec![]), &Value::Vector(vec![])));
        assert!(equal(
            &Value::List(vec![Value::Boolean(true)]),
            &Value::Vector(vec![Value::Boolean(true)])
        ));
        assert!(!equal(
            &Value::List(vec![Value::Boolean(true)]),
            &Value::Vector(vec![Value::Boolean(false)])
        ));
        assert!(!equal(
            &Value::List(vec![Value::Boolean(true)]),
            &Value::Vector(vec![Value::Boolean(true), Value::Boolean(true)])
        ));
    }

    #[test]
    fn test_parsing_string() {
        assert_eq!(
            Value::String("ꪪ".to_string()),
            parse("\"\\uAAAA\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_string_nested() {
        assert_eq!(
            Value::Vector(vec![Value::String("ꪪ".to_string())]),
            parse("[\"\\uAAAA\"]").unwrap()
        )
    }

    #[test]
    fn test_parsing_multiline_string() {
        assert_eq!(
            Value::String("abc\n    \ndef    \n".to_string()),
            parse("\"abc\n    \ndef    \n\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_string_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::String("abc".to_string()),
                Value::String("def".to_string())
            )])),
            parse("{\"abc\" \"def\"}").unwrap()
        );

        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::String("abc".to_string()),
                Value::String("def".to_string())
            )])),
            parse("{\"abc\"\"def\"}").unwrap()
        )
    }

    #[test]
    fn test_parsing_inst() {
        assert_eq!(
            Value::Inst(DateTime::parse_from_rfc3339("1985-04-12T23:20:50.52Z").unwrap()),
            parse("#inst\"1985-04-12T23:20:50.52Z\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_uuid() {
        assert_eq!(
            Value::Uuid(Uuid::parse_str("f81d4fae-7dec-11d0-a765-00a0c91e6bf6").unwrap()),
            parse("#uuid\"f81d4fae-7dec-11d0-a765-00a0c91e6bf6\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_symbol() {
        assert_eq!(
            Value::Vector(vec![
                Value::Symbol(Symbol::from_name("a")),
                Value::Symbol(Symbol::from_name("abc")),
                Value::Symbol(Symbol::from_namespace_and_name("abc", "def")),
                Value::Symbol(Symbol::from_name("->")),
                Value::Symbol(Symbol::from_name("/")),
                Value::Symbol(Symbol::from_namespace_and_name("my.org", "stuff")),
            ]),
            parse("[ a  abc abc/def -> / my.org/stuff ]").unwrap()
        );
    }

    #[test]
    fn test_parsing_symbol_errs() {
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtBeginningOfSymbol),
            parse("/abc")
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfSymbol),
            parse("abc/")
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfSymbol),
            parse("abc/ ")
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfSymbol),
            parse("abc/ []")
        );
        assert_eq!(
            Err(ParserError::CannotHaveMoreThanOneSlashInSymbol),
            parse("a/b/c")
        );
    }

    #[test]
    fn test_parsing_keyword() {
        assert_eq!(
            Value::Vector(vec![
                Value::Keyword(Keyword::from_name("a")),
                Value::Keyword(Keyword::from_name("abc")),
                Value::Keyword(Keyword::from_namespace_and_name("abc", "def")),
                Value::Keyword(Keyword::from_name("->")),
                Value::Keyword(Keyword::from_name("/")),
                Value::Keyword(Keyword::from_namespace_and_name("my.org", "stuff")),
            ]),
            parse("[ :a  :abc :abc/def :-> :/ :my.org/stuff ]").unwrap()
        );
    }

    #[test]
    fn test_parsing_keyword_errs() {
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtBeginningOfKeyword),
            parse(":/abc")
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfKeyword),
            parse(":abc/")
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfKeyword),
            parse(":abc/ ")
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfKeyword),
            parse(":abc/ []")
        );
        assert_eq!(
            Err(ParserError::CannotHaveMoreThanOneSlashInKeyword),
            parse(":a/b/c")
        );
        assert_eq!(Err(ParserError::InvalidKeyword), parse("::namespaced"));
    }

    #[test]
    fn test_parse_set() {
        assert_eq!(
            Value::Set(BTreeSet::from_iter(
                vec![
                    Value::Keyword(Keyword::from_name("abc")),
                    Value::Keyword(Keyword::from_name("def")),
                    Value::Keyword(Keyword::from_namespace_and_name("ghi", "jkl"))
                ]
                .into_iter()
            )),
            parse("#{:abc :def :ghi/jkl }").unwrap()
        );

        assert_eq!(
            Err(ParserError::DuplicateValueInSet {
                value: Value::Symbol(Symbol::from_name("a"))
            }),
            parse("#{a b c d e f a g h}")
        )
    }

    #[test]
    fn test_set_equals() {
        assert!(equal(
            &Value::Set(BTreeSet::from_iter(
                vec![
                    Value::Keyword(Keyword::from_name("def")),
                    Value::Keyword(Keyword::from_namespace_and_name("ghi", "jkl")),
                    Value::Keyword(Keyword::from_name("abc"))
                ]
                .into_iter()
            )),
            &Value::Set(BTreeSet::from_iter(
                vec![
                    Value::Keyword(Keyword::from_name("abc")),
                    Value::Keyword(Keyword::from_name("def")),
                    Value::Keyword(Keyword::from_namespace_and_name("ghi", "jkl"))
                ]
                .into_iter()
            ))
        ))
    }
    #[test]
    fn test_example_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![
                (
                    Value::Keyword(Keyword::from_namespace_and_name("person", "name")),
                    Value::String("Joe".to_string())
                ),
                (
                    Value::Keyword(Keyword::from_namespace_and_name("person", "parent")),
                    Value::String("Bob".to_string())
                ),
                (
                    Value::Keyword(Keyword::from_name("ssn")),
                    Value::String("123".to_string())
                ),
                (
                    Value::Symbol(Symbol::from_name("friends")),
                    Value::Vector(vec![
                        Value::String("sally".to_string()),
                        Value::String("john".to_string()),
                        Value::String("linda".to_string())
                    ])
                ),
                (
                    Value::String("other".to_string()),
                    Value::Map(BTreeMap::from_iter(vec![(
                        Value::Keyword(Keyword::from_name("stuff")),
                        Value::Keyword(Keyword::from_name("here"))
                    )]))
                )
            ])),
            parse(
                "\
            {:person/name \"Joe\"\
             :person/parent \"Bob\"\
             :ssn \"123\"\
             friends [\"sally\" \"john\" \"linda\"]\
             \"other\" {:stuff :here}}"
            )
            .unwrap()
        )
    }

    #[test]
    fn test_basic_keyword_and_symbol() {
        assert!(equal(&parse("name").unwrap(), &parse("name").unwrap()));
        assert!(equal(
            &parse("person/name").unwrap(),
            &parse("person/name").unwrap()
        ));
        assert!(equal(&parse(":name").unwrap(), &parse(":name").unwrap()));
        assert!(equal(
            &parse(":person/name").unwrap(),
            &parse(":person/name").unwrap()
        ));

        // Had an issue with whitespace
        assert!(equal(&parse("name ").unwrap(), &parse("name ").unwrap()));
        assert!(equal(
            &parse("person/name ").unwrap(),
            &parse("person/name ").unwrap()
        ));
        assert!(equal(&parse(":name ").unwrap(), &parse(":name ").unwrap()));
        assert!(equal(
            &parse(":person/name ").unwrap(),
            &parse(":person/name ").unwrap()
        ));
    }

    #[test]
    fn test_complex_equals() {
        assert!(equal(
            &parse(
                "\
            {:person/parent \"Bob\"\
             :person/name \"Joe\"\
             :ssn \"123\"\
             friends [\"sally\" \"john\" \"linda\"]\
             \"other\" {:stuff :here}}"
            )
            .unwrap(),
            &parse(
                "\
            {:person/name \"Joe\"\
             :person/parent \"Bob\"\
             :ssn \"123\"\
             friends [\"sally\" \"john\" \"linda\"]\
             \"other\" {:stuff :here}}"
            )
            .unwrap()
        ))
    }

    #[test]
    fn test_nil_false_true() {
        assert_eq!(
            Value::List(vec![
                Value::Nil,
                Value::Boolean(false),
                Value::Boolean(true)
            ]),
            parse("(nil false true)").unwrap()
        )
    }

    #[test]
    fn test_parse_char() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![
                (Value::Character(' '), Value::Character('z')),
                (Value::Character('a'), Value::Character('\n')),
                (Value::Character('b'), Value::Character('\r')),
                (Value::Character('r'), Value::Character('c')),
                (Value::Character('\t'), Value::Character('d')),
            ])),
            parse("{\\space \\z\\a \\newline \\b \\return \\r \\c \\tab \\d}").unwrap()
        )
    }

    #[test]
    fn test_parse_int() {
        assert_eq!(Value::Integer(123), parse("123").unwrap())
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(Value::Float(OrderedFloat(12.1)), parse("12.1").unwrap())
    }

    #[test]
    fn test_parse_neg_int() {
        assert_eq!(Value::Integer(-123), parse("-123").unwrap())
    }

    #[test]
    fn test_parse_neg_float() {
        assert_eq!(Value::Float(OrderedFloat(-12.1)), parse("-12.1").unwrap())
    }

    #[test]
    fn test_parse_pos_int() {
        assert_eq!(Value::Integer(123), parse("+123").unwrap())
    }

    #[test]
    fn test_parse_pos_float() {
        assert_eq!(Value::Float(OrderedFloat(12.1)), parse("+12.1").unwrap())
    }

    #[test]
    fn test_parse_zero() {
        assert_eq!(Value::Integer(0), parse("+0").unwrap(),);
        assert_eq!(Value::Integer(0), parse("0").unwrap(),);
        assert_eq!(Value::Integer(0), parse("-0").unwrap(),);
    }

    #[test]
    fn test_parse_zero_float() {
        assert_eq!(Value::Float(OrderedFloat(0f64)), parse("+0.").unwrap());
        assert_eq!(Value::Float(OrderedFloat(0f64)), parse("0.").unwrap());
        assert_eq!(Value::Float(OrderedFloat(0f64)), parse("-0.").unwrap());
    }

    #[test]
    fn test_parse_e() {
        assert_eq!(Value::Float(OrderedFloat(1000.0)), parse("10e+2").unwrap());
        assert_eq!(Value::Float(OrderedFloat(1200.0)), parse("12e+2").unwrap());
        assert_eq!(Value::Float(OrderedFloat(1200.0)), parse("12e2").unwrap());
        assert_eq!(Value::Float(OrderedFloat(1200.0)), parse("12E2").unwrap());
        assert_eq!(Value::Float(OrderedFloat(5200.0)), parse("52E+2").unwrap());
        assert_eq!(Value::Float(OrderedFloat(1.2)), parse("120e-2").unwrap());
        assert_eq!(Value::Float(OrderedFloat(1.2)), parse("120E-2").unwrap());
        assert_eq!(
            Value::Float(OrderedFloat(1422141241242142142141241.124)),
            parse("1422141241242142142141241.124E0").unwrap()
        );
    }

    #[test]
    fn test_parse_bigint() {
        assert_eq!(Value::BigInt(BigInt::from(123)), parse("123N").unwrap());
    }

    #[test]
    fn test_parse_bigdec() {
        assert_eq!(Value::BigDec(BigDecimal::from(123)), parse("123M").unwrap());
    }

    #[test]
    fn test_with_comments() {
        assert_eq!(
            Value::List(
                vec![
                    Value::Integer(1),
                    Value::Integer(2),
                    Value::Integer(3),
                    Value::String("abc".to_string()),
                    Value::Vector(
                        vec![
                            Value::Symbol(Symbol::from_name("a")),
                            Value::Symbol(Symbol::from_namespace_and_name("b", "qq")),
                            Value::Symbol(Symbol::from_name("c")),
                            Value::Map(BTreeMap::from_iter(vec![
                                (Value::Integer(12), Value::Float(OrderedFloat(34.5))),
                                (Value::Keyword(Keyword::from_name("a")),
                                 Value::Symbol(Symbol::from_name("b")))
                            ]))
                        ]
                    )
                ]

            ),
            parse("( 1 2 3 \"abc\"\n;; so here is where we do some wacky stuff \n [a b/qq ; and then here\n c \n \n;; aaa\n{12 34.5 :a \n;;aaadeafaef\nb}])").unwrap()
        );
    }

    #[test]
    fn test_round_trips() {
        let v1 = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
            Value::String("abc".to_string()),
            Value::Vector(vec![
                Value::Symbol(Symbol::from_name("a")),
                Value::Symbol(Symbol::from_namespace_and_name("b", "qq")),
                Value::Symbol(Symbol::from_name("c")),
                Value::Map(BTreeMap::from_iter(vec![
                    (Value::Integer(12), Value::Float(OrderedFloat(34.5))),
                    (
                        Value::Keyword(Keyword::from_name("a")),
                        Value::Symbol(Symbol::from_name("b")),
                    ),
                ])),
            ]),
        ]);
        let v2 = Value::Map(BTreeMap::from_iter(vec![
            (
                Value::Keyword(Keyword::from_namespace_and_name("person", "name")),
                Value::String("Joe".to_string()),
            ),
            (
                Value::Keyword(Keyword::from_namespace_and_name("person", "parent")),
                Value::String("Bob".to_string()),
            ),
            (
                Value::Keyword(Keyword::from_name("ssn")),
                Value::String("123".to_string()),
            ),
            (
                Value::Symbol(Symbol::from_name("friends")),
                Value::Vector(vec![
                    Value::String("sally".to_string()),
                    Value::String("john".to_string()),
                    Value::String("linda".to_string()),
                ]),
            ),
            (
                Value::String("other".to_string()),
                Value::Map(BTreeMap::from_iter(vec![(
                    Value::Keyword(Keyword::from_name("stuff")),
                    Value::Keyword(Keyword::from_name("here")),
                )])),
            ),
            (
                Value::Inst(DateTime::parse_from_rfc3339("1985-04-12T23:20:50.52Z").unwrap()),
                Value::Set(BTreeSet::from_iter(
                    vec![
                        Value::TaggedElement(
                            Symbol::from_namespace_and_name("person", "ssn"),
                            Box::new(Value::String("123".to_string())),
                        ),
                        Value::Nil,
                        Value::Boolean(false),
                        Value::List(vec![
                            Value::Integer(1),
                            Value::Float(OrderedFloat(2.0)),
                            Value::BigDec(BigDecimal::from((BigInt::from(4), 9))),
                            Value::BigInt(BigInt::from(4)),
                        ]),
                    ]
                    .into_iter(),
                )),
            ),
        ]));
        assert_eq!(emit(&v1), emit(&parse(&emit(&v1)).unwrap()));
        assert_eq!(emit(&v2), emit(&parse(&emit(&v2)).unwrap()));

        assert_eq!(
            parse(&emit(&v1)).unwrap(),
            parse(&emit(&parse(&emit(&v1)).unwrap())).unwrap()
        );
        assert_eq!(
            parse(&emit(&v2)).unwrap(),
            parse(&emit(&parse(&emit(&v2)).unwrap())).unwrap()
        );
    }

    #[test]
    fn test_big_vector() {
        let mut vals: Vec<Value> = vec![];
        for i in 0..100000 {
            vals.push(Value::Integer(i))
        }
        let ser = emit(&Value::Vector(vals.clone()));
        // shouldn't stack overflow
        assert_eq!(parse(&ser).unwrap(), Value::Vector(vals));
    }

    #[test]
    fn test_big_list() {
        let mut vals: Vec<Value> = vec![];
        for i in 0..100000 {
            vals.push(Value::Integer(i))
        }
        let ser = emit(&Value::List(vals.clone()));
        assert_eq!(parse(&ser).unwrap(), Value::List(vals));
    }

    #[test]
    fn test_big_set() {
        let mut vals: Vec<Value> = vec![];
        for i in 0..100000 {
            vals.push(Value::Integer(i))
        }
        let ser = emit(&Value::Set(BTreeSet::from_iter(vals.clone())));
        assert_eq!(parse(&ser).unwrap(), Value::Set(BTreeSet::from_iter(vals)));
    }

    #[test]
    fn test_big_map() {
        let mut vals: Vec<(Value, Value)> = vec![];
        for i in 0..10000 {
            vals.push((
                Value::Keyword(Keyword::from_name(&i.to_string())),
                Value::Integer(i),
            ))
        }
        let ser = emit(&Value::Map(BTreeMap::from_iter(vals.clone())));
        assert_eq!(parse(&ser).unwrap(), Value::Map(BTreeMap::from_iter(vals)));
    }

    #[test]
    fn test_two_colons() {
        assert_eq!(Err(ParserError::InvalidKeyword), parse("::"))
    }

    #[test]
    fn test_row_col_tracking() {
        assert_eq!(
            Err(ParserError::WithContext {
                context: Vec::new(),
                row_col: RowCol { row: 6, col: 1 },
                error: Box::new(ParserError::InvalidKeyword)
            }),
            parse_with_options(
                "    ::",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );
        assert_eq!(
            Err(ParserError::WithContext {
                context: Vec::new(),
                row_col: RowCol { row: 0, col: 1 },
                error: Box::new(ParserError::EmptyInput)
            }),
            parse_with_options(
                "",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );
        assert_eq!(
            Err(ParserError::WithContext {
                context: Vec::new(),
                row_col: RowCol { row: 5, col: 1 },
                error: Box::new(ParserError::InvalidKeyword)
            }),
            parse_with_options(
                "   ::",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );
        assert_eq!(
            Err(ParserError::WithContext {
                context: Vec::new(),
                row_col: RowCol { row: 2, col: 3 },
                error: Box::new(ParserError::InvalidKeyword)
            }),
            parse_with_options(
                "   \n\n::",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );
    }

    #[test]
    fn test_context_maintaining() {
        assert_eq!(
            Err(ParserError::WithContext {
                context: vec![Context::ParsingVector(RowCol { row: 2, col: 1 })],
                row_col: RowCol { row: 10, col: 1 },
                error: Box::new(ParserError::InvalidKeyword)
            }),
            parse_with_options(
                " [ 1 2 ::a]",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );

        assert_eq!(
            Err(ParserError::WithContext {
                context: vec![
                    Context::ParsingList(RowCol { row: 2, col: 1 }),
                    Context::ParsingSet(RowCol { row: 2, col: 2 }),
                    Context::ParsingMap(RowCol { row: 1, col: 3 }),
                    Context::ParsingVector(RowCol { row: 3, col: 3 }),
                ],
                row_col: RowCol { row: 5, col: 5 },
                error: Box::new(ParserError::InvalidKeyword)
            }),
            parse_with_options(
                " ( a b c \n#{ \n{ [ \n1 2 4\n  ::a  \n3]  3} } )",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );
        assert_eq!(
            Err(ParserError::WithContext {
                context: vec![Context::ParsingList(RowCol { row: 2, col: 1 })],
                row_col: RowCol { row: 8, col: 1 },
                error: Box::new(ParserError::UnexpectedEndOfInput)
            }),
            parse_with_options(
                " ( [] {}",
                ParserOptions {
                    track_line_numbers: true,
                    ..ParserOptions::default()
                }
            )
        );
    }

    #[test]
    fn test_namespaced_maps() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![
                (
                    Value::Keyword(Keyword::from_namespace_and_name("apple.tree", "c")),
                    Value::Integer(1)
                ),
                (Value::String("a".to_string()), Value::Integer(2)),
                (
                    Value::Keyword(Keyword::from_namespace_and_name("apple.tree", "d")),
                    Value::Integer(3)
                ),
                (
                    Value::Keyword(Keyword::from_namespace_and_name("d.e", "f")),
                    Value::Keyword(Keyword::from_name("a"))
                ),
            ])),
            parse_with_options(
                "#:apple.tree{:c 1 \"a\" 2 :d 3 :d.e/f :a}",
                ParserOptions {
                    allow_namespaced_map_syntax: true,
                    ..ParserOptions::default()
                }
            )
            .unwrap()
        )
    }
}
