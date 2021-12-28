use bigdecimal::{BigDecimal, ParseBigDecimalError};
use chrono::FixedOffset;
use internship::IStr;
use itertools::Itertools;
use num_bigint::{BigInt, ParseBigIntError};
use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::iter::{FromIterator, Peekable};
use std::num::{ParseFloatError, ParseIntError};
use std::str::{Chars, FromStr};
use thiserror::Error;
use uuid::Uuid;

/// A keyword, as described in EDN data model, is identifier which should
/// "designate itself".
///
/// Because its contents are interned, cloning and comparisons should be relatively
/// cheap operations and construction relatively expensive.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Keyword {
    namespace: Option<IStr>,
    name: IStr,
}

impl Keyword {
    /// The namespace of the keyword, if there is one.
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_ref().map(|s| s.as_str())
    }

    /// The name of the keyword.
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
/// a keyword should give valid EDN.
impl Display for Keyword {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.namespace {
            Some(ns) => write!(f, ":{}/{}", ns, self.name),
            None => write!(f, ":{}", self.name),
        }
    }
}

/// A symbol, as described in EDN data model, is an identifier.
///
/// Because its contents are interned, cloning and comparisons should be relatively
/// cheap operations and construction relatively expensive.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Symbol {
    namespace: Option<IStr>,
    name: IStr,
}

impl Symbol {
    /// The namespace of the symbol, if there is one.
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_ref().map(|s| s.as_str())
    }

    /// The name of the symbol.
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
/// a symbol should give valid EDN.
impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.namespace {
            Some(ns) => write!(f, "{}/{}", ns, self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

/// An EDN Value.
#[derive(Debug, Clone, Ord, PartialOrd, Eq)]
pub enum Value {
    /// `nil`. Analogous to `null`, nothing, zip, zilch, and nada.
    ///
    /// <https://github.com/edn-format/edn#nil>
    Nil,
    /// A boolean value
    ///
    /// <https://github.com/edn-format/edn#booleans>
    Boolean(bool),
    /// A single character.
    ///
    /// <https://github.com/edn-format/edn#characters>
    Character(char),
    /// A string. Used to represent textual data.
    ///
    /// <https://github.com/edn-format/edn#strings>
    String(String),
    /// A symbol. Used to represent identifiers.
    ///
    /// <https://github.com/edn-format/edn#symbols>
    Symbol(Symbol),
    /// A keyword. An identifier that designates itself. For typical use,
    /// these should be used for keys in [Value::Map] instead of [Value::String].
    ///
    /// <https://github.com/edn-format/edn#keywords>
    Keyword(Keyword),
    /// A signed integer
    ///
    /// <https://github.com/edn-format/edn#integers>
    Integer(i64),
    /// A floating point number with 64 bit precision.
    ///
    /// <https://github.com/edn-format/edn#floating-point-numbers>
    Float(OrderedFloat<f64>),
    /// An integer with arbitrary precision
    ///
    /// <https://github.com/edn-format/edn#integers>
    BigInt(BigInt),
    /// A decimal number with exact precision
    ///
    /// <https://github.com/edn-format/edn#floating-point-numbers>
    BigDec(BigDecimal),
    /// A list of values.
    ///
    /// <https://github.com/edn-format/edn#lists>
    List(Vec<Value>),
    /// A vector of values. The major difference between this and a [Value::List]
    /// is that a vector is guaranteed by the spec to support random access of
    /// elements. In this implementation the semantic difference is maintained,
    /// but the underlying data structure for both is a [Vec], which supports random
    /// access.
    ///
    /// <https://github.com/edn-format/edn#vectors>
    Vector(Vec<Value>),
    /// A collection of associations between keys and values. Supports any EDN value as a
    /// key. No semantics should be associated with the order in which the pairs appear.
    ///
    /// <https://github.com/edn-format/edn#maps>
    Map(BTreeMap<Value, Value>),
    /// A collection of unique values. No semantics should be associated with the order the
    /// items appear.
    ///
    /// <https://github.com/edn-format/edn#sets>
    Set(BTreeSet<Value>),
    /// An instant in time. Represented as a tagged element with tag `inst` and value
    /// a [RFC-3339](https://www.ietf.org/rfc/rfc3339.txt) formatted string.
    ///
    /// <https://github.com/edn-format/edn#inst-rfc-3339-format>
    Inst(chrono::DateTime<FixedOffset>),
    /// A UUID. Represented as a tagged element with tag `uuid` and value
    /// a canonical UUID string.
    ///
    /// <https://github.com/edn-format/edn#uuid-f81d4fae-7dec-11d0-a765-00a0c91e6bf6>
    Uuid(Uuid),
    /// A tagged element. This can be used to encode any kind of data as a distinct
    /// readable element, with semantics determined by the reader and writer.
    ///
    /// Overriding the behavior of elements tagged with `inst` and `uuid` is not supported
    /// in this implementation.
    ///
    /// <https://github.com/edn-format/edn#tagged-elements>
    TaggedElement(Symbol, Box<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        equal(self, other)
    }
}

impl From<()> for Value {
    fn from(_: ()) -> Self {
        Value::Nil
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        opt.map(|o| o.into()).unwrap_or(Value::Nil)
    }
}

impl From<char> for Value {
    fn from(c: char) -> Self {
        Value::Character(c)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<Symbol> for Value {
    fn from(sym: Symbol) -> Self {
        Value::Symbol(sym)
    }
}

impl From<Keyword> for Value {
    fn from(kw: Keyword) -> Self {
        Value::Keyword(kw)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Integer(i)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(OrderedFloat(f))
    }
}

impl From<OrderedFloat<f64>> for Value {
    fn from(of: OrderedFloat<f64>) -> Self {
        Value::Float(of)
    }
}

impl From<BigInt> for Value {
    fn from(bi: BigInt) -> Self {
        Value::BigInt(bi)
    }
}

impl From<BigDecimal> for Value {
    fn from(bd: BigDecimal) -> Self {
        Value::BigDec(bd)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Boolean(b)
    }
}

impl From<chrono::DateTime<FixedOffset>> for Value {
    fn from(date: chrono::DateTime<FixedOffset>) -> Self {
        Value::Inst(date)
    }
}

impl From<Uuid> for Value {
    fn from(uuid: Uuid) -> Self {
        Value::Uuid(uuid)
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
    InvalidSymbol,

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

    #[error("Cannot have more a colon in a symbol")]
    CannotHaveColonInSymbol,

    #[error("Cannot have slash at the beginning of symbol")]
    CannotHaveSlashAtBeginningOfKeyword,

    #[error("Cannot have slash at the end of symbol")]
    CannotHaveSlashAtEndOfKeyword,

    #[error("Cannot have more than one slash in a symbol")]
    CannotHaveMoreThanOneSlashInKeyword,

    #[error("Cannot have more a colon in a keyword")]
    CannotHaveColonInKeyword,

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
    ExtraInput { parsed_value: Value },
}

#[derive(Debug, Error, PartialEq)]
pub struct ParserErrorWithContext {
    /// Vector holding the context that the parser was working in.
    ///
    /// Intended to be useful for giving hints to a user about how to fix their input.
    pub context: Vec<Context>,
    /// The row where the error was detected. 1 indexed. 0 if no characters on the line have been read.
    pub row: usize,
    /// The col where the error was detected. 1 indexed. Starts at 1.
    pub col: usize,
    /// The error that the parser ran into.
    pub error: ParserError,
}

impl Display for ParserErrorWithContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}): {}", self.row, self.col, self.error)
    }
}

#[derive(Debug)]
enum ParserState {
    Begin,
    ParsingList { values_so_far: Vec<Value> },
    ParsingVector { values_so_far: Vec<Value> },
    ParsingMap { values_so_far: Vec<Value> },
    ParsingSet { values_so_far: Vec<Value> },
    ParsingAtom { built_up: Vec<char> },
    ParsingString { built_up: String },
    ParsingCharacter,
    SelectingDispatch,
}

/// Commas are considered whitespace for EDN
fn is_whitespace(c: char) -> bool {
    c.is_whitespace() || c == ','
}

fn is_allowed_atom_character(c: char) -> bool {
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
        || c == '/'
        || c == ':'
        || c.is_alphabetic()
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
            vals1 == vals2
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
struct RowCol {
    /// The row. 1 indexed. 0 if no characters on the line have been read.
    row: usize,
    /// The col. 1 indexed. Starts at 1.
    col: usize,
}

impl RowCol {
    fn new() -> RowCol {
        RowCol { row: 0, col: 1 }
    }

    fn chomp(&mut self, c: char) {
        if c == '\n' {
            self.row = 0;
            self.col += 1;
        } else {
            self.row += 1;
        }
    }
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

    fn start_parsing_atom(&mut self);

    fn stop_parsing_current(&mut self);

    fn advance_one_char(&mut self, c: char);
}

/// An element of context about what the parser was doing when an error was detected.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Context {
    /// The parser started parsing a vector at the given row and col.
    ParsingVector { row: usize, col: usize },
    /// The parser started parsing a vector at the given row and col.
    ParsingList { row: usize, col: usize },
    /// The parser started parsing a vector at the given row and col.
    ParsingMap { row: usize, col: usize },
    /// The parser started parsing a vector at the given row and col.
    ParsingSet { row: usize, col: usize },
    /// The parser started parsing a vector at the given row and col.
    ParsingString { row: usize, col: usize },
    /// The parser started parsing an "atom" at the given row and col.
    ///
    /// An "atom" might be a symbol, keyword, number, true, false, or nil.
    /// At the point the parser records this information it does not know which.
    ParsingAtom { row: usize, col: usize },
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
            row_col: RowCol::new(),
        }
    }
}

impl ParseObserver for ContextStackerObserver {
    fn start_parsing_vector(&mut self) {
        self.context.push(Context::ParsingVector {
            row: self.row_col.row,
            col: self.row_col.col,
        });
    }

    fn start_parsing_list(&mut self) {
        self.context.push(Context::ParsingList {
            row: self.row_col.row,
            col: self.row_col.col,
        });
    }

    fn start_parsing_map(&mut self) {
        self.context.push(Context::ParsingMap {
            row: self.row_col.row,
            col: self.row_col.col,
        });
    }

    fn start_parsing_set(&mut self) {
        self.context.push(Context::ParsingSet {
            row: self.row_col.row,
            col: self.row_col.col,
        });
    }

    fn start_parsing_string(&mut self) {
        self.context.push(Context::ParsingString {
            row: self.row_col.row,
            col: self.row_col.col,
        });
    }

    fn start_parsing_atom(&mut self) {
        self.context.push(Context::ParsingAtom {
            row: self.row_col.row,
            col: self.row_col.col,
        });
    }

    fn stop_parsing_current(&mut self) {
        self.context.pop();
    }

    fn advance_one_char(&mut self, c: char) {
        self.row_col.chomp(c)
    }
}

fn interpret_atom(atom: &[char]) -> Result<Value, ParserError> {
    let char_slice_to_str = |chars: &[char]| chars.iter().copied().collect::<String>();
    let starts_like_number = |chars: &[char]| {
        !chars.is_empty()
            && (chars[0].is_numeric()
                || (['+', '-'].contains(&chars[0]) && chars.len() >= 2 && chars[1].is_numeric()))
    };
    match atom {
        &[] => Err(ParserError::UnexpectedEndOfInput),
        &['n', 'i', 'l'] => Ok(Value::Nil),
        &['t', 'r', 'u', 'e'] => Ok(Value::Boolean(true)),
        &['f', 'a', 'l', 's', 'e'] => Ok(Value::Boolean(false)),
        &[':'] => Err(ParserError::InvalidKeyword),
        &[':', '/'] => Ok(Value::Keyword(Keyword::from_name("/"))),
        &[':', '/', ..] => Err(ParserError::CannotHaveSlashAtBeginningOfKeyword),
        &[':', .., '/'] => Err(ParserError::CannotHaveSlashAtEndOfKeyword),
        &['/'] => Ok(Value::Symbol(Symbol::from_name("/"))),
        &['/', ..] => Err(ParserError::CannotHaveSlashAtBeginningOfSymbol),
        &[.., '/'] => Err(ParserError::CannotHaveSlashAtEndOfSymbol),
        &[':', ref rest @ ..] => {
            if rest.contains(&':') {
                Err(ParserError::CannotHaveColonInKeyword)
            } else {
                let split: Vec<_> = rest.split(|c| *c == '/').collect();
                match split[..] {
                    [name] => Ok(Value::Keyword(Keyword::from_name(&char_slice_to_str(name)))),
                    [namespace, name] => {
                        if starts_like_number(namespace) {
                            Err(ParserError::InvalidKeyword)
                        } else {
                            Ok(Value::Keyword(Keyword::from_namespace_and_name(
                                &char_slice_to_str(namespace),
                                &char_slice_to_str(name),
                            )))
                        }
                    }
                    _ => Err(ParserError::CannotHaveMoreThanOneSlashInKeyword),
                }
            }
        }
        chars => {
            if chars.contains(&':') {
                Err(ParserError::CannotHaveColonInSymbol)
            } else {
                let split: Vec<_> = chars.split(|c| *c == '/').collect();
                match split[..] {
                    [name] => {
                        if starts_like_number(name) {
                            if name.ends_with(&['M'])
                                && name.iter().filter(|c| **c == 'M').count() == 1
                            {
                                Ok(Value::BigDec(
                                    str::parse::<BigDecimal>(&char_slice_to_str(
                                        &chars[..chars.len() - 1],
                                    ))
                                    .map_err(|err| {
                                        ParserError::BadBigDec {
                                            parsing: char_slice_to_str(chars),
                                            encountered: err,
                                        }
                                    })?,
                                ))
                            } else if name.contains(&'.')
                                || name.contains(&'e')
                                || name.contains(&'E')
                            {
                                Ok(Value::Float(OrderedFloat(
                                    str::parse::<f64>(&char_slice_to_str(chars)).map_err(
                                        |err| ParserError::BadFloat {
                                            parsing: char_slice_to_str(chars),
                                            encountered: err,
                                        },
                                    )?,
                                )))
                            } else if name != ['0'] && (name.starts_with(&['0']))
                                || (name != ['+', '0']
                                    && (name.len() > 1
                                        && name.starts_with(&['+'])
                                        && name[1..].starts_with(&['0'])))
                                || (name != ['-', '0']
                                    && (name.len() > 1
                                        && name.starts_with(&['-'])
                                        && name[1..].starts_with(&['0'])))
                            {
                                // Only ints are subject to this restriction it seems
                                Err(ParserError::OnlyZeroCanStartWithZero)
                            } else if name.ends_with(&['N'])
                                && name.iter().filter(|c| **c == 'N').count() == 1
                            {
                                Ok(Value::BigInt(
                                    str::parse::<BigInt>(&char_slice_to_str(
                                        &chars[..chars.len() - 1],
                                    ))
                                    .map_err(|err| {
                                        ParserError::BadBigInt {
                                            parsing: char_slice_to_str(chars),
                                            encountered: err,
                                        }
                                    })?,
                                ))
                            } else {
                                Ok(Value::Integer(
                                    str::parse::<i64>(&char_slice_to_str(chars)).map_err(
                                        |err| ParserError::BadInt {
                                            parsing: char_slice_to_str(chars),
                                            encountered: err,
                                        },
                                    )?,
                                ))
                            }
                        } else {
                            Ok(Value::Symbol(Symbol::from_name(&char_slice_to_str(name))))
                        }
                    }
                    [namespace, name] => {
                        if starts_like_number(namespace) {
                            Err(ParserError::InvalidSymbol)
                        } else {
                            Ok(Value::Symbol(Symbol::from_namespace_and_name(
                                &char_slice_to_str(namespace),
                                &char_slice_to_str(name),
                            )))
                        }
                    }
                    _ => Err(ParserError::CannotHaveMoreThanOneSlashInSymbol),
                }
            }
        }
    }
}

/// Likely suboptimal parsing. Focus for now is just on getting correct results.
fn parse_helper<Observer: ParseObserver, Iter: Iterator<Item = char>>(
    s: &mut Peekable<Iter>,
    mut parser_state: ParserState,
    observer: &mut Observer,
    opts: &ParserOptions,
) -> Result<Value, ParserError> {
    'parsing: loop {
        // Strip out comments
        match parser_state {
            ParserState::ParsingString { .. } => {}
            _ => {
                if let Some(';') = s.peek() {
                    loop {
                        match s.next() {
                            Some('\n') | None => {
                                observer.advance_one_char('\n');
                                break;
                            }
                            Some(c) => {
                                observer.advance_one_char(c);
                            }
                        }
                    }
                }
            }
        };
        match &mut parser_state {
            ParserState::Begin => match s.next() {
                None => return Err(ParserError::EmptyInput),
                Some(c) => {
                    observer.advance_one_char(c);
                    if is_whitespace(c) {
                        parser_state = ParserState::Begin;
                    } else if c == '(' {
                        observer.start_parsing_list();
                        parser_state = ParserState::ParsingList {
                            values_so_far: vec![],
                        };
                    } else if c == '[' {
                        observer.start_parsing_vector();
                        parser_state = ParserState::ParsingVector {
                            values_so_far: vec![],
                        };
                    } else if c == '{' {
                        observer.start_parsing_map();
                        parser_state = ParserState::ParsingMap {
                            values_so_far: vec![],
                        };
                    } else if c == '"' {
                        observer.start_parsing_string();
                        parser_state = ParserState::ParsingString {
                            built_up: "".to_string(),
                        };
                    } else if c == '\\' {
                        parser_state = ParserState::ParsingCharacter;
                    } else if c == '#' {
                        parser_state = ParserState::SelectingDispatch;
                    } else if is_allowed_atom_character(c) || c == ':' {
                        let built_up = vec![c];
                        observer.start_parsing_atom();
                        parser_state = ParserState::ParsingAtom { built_up };
                    } else {
                        return Err(ParserError::UnexpectedCharacter(c));
                    }
                }
            },

            ParserState::ParsingList {
                ref mut values_so_far,
            } => {
                let next = s.peek();
                match next {
                    None => return Err(ParserError::UnexpectedEndOfInput),
                    Some(c) => {
                        if is_whitespace(*c) {
                            observer.advance_one_char(*c);
                            s.next();
                        } else if *c == ')' {
                            observer.advance_one_char(*c);
                            s.next();
                            observer.stop_parsing_current();
                            return Ok(Value::List(values_so_far.clone()));
                        } else {
                            let value = parse_helper(s, ParserState::Begin, observer, opts)?;
                            values_so_far.push(value);
                        }
                    }
                }
            }

            // Almost total duplicate of ParsingList
            ParserState::ParsingVector {
                ref mut values_so_far,
            } => match s.peek() {
                None => {
                    return Err(ParserError::UnexpectedEndOfInput);
                }
                Some(c) => {
                    if is_whitespace(*c) {
                        observer.advance_one_char(*c);
                        s.next();
                    } else if *c == ']' {
                        observer.stop_parsing_current();
                        observer.advance_one_char(*c);
                        s.next();
                        return Ok(Value::Vector(values_so_far.clone()));
                    } else {
                        let value = parse_helper(s, ParserState::Begin, observer, opts)?;
                        values_so_far.push(value);
                    }
                }
            },

            ParserState::ParsingMap {
                ref mut values_so_far,
            } => {
                match s.peek() {
                    None => {
                        return Err(ParserError::UnexpectedEndOfInput);
                    }
                    Some(c) => {
                        if is_whitespace(*c) {
                            observer.advance_one_char(*c);
                            s.next();
                        } else if *c == '}' {
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
                                        return Err(ParserError::DuplicateKeyInMap {
                                            value: k.clone(),
                                        });
                                    }
                                    seen.insert(k);
                                }
                                let value = Value::Map(BTreeMap::from_iter(entries));

                                observer.stop_parsing_current();
                                observer.advance_one_char(*c);
                                s.next();
                                return Ok(value);
                            }
                        } else {
                            let value = parse_helper(s, ParserState::Begin, observer, opts)?;
                            values_so_far.push(value);
                        }
                    }
                }
            }

            ParserState::ParsingSet {
                ref mut values_so_far,
            } => match s.peek() {
                None => {
                    return Err(ParserError::UnexpectedEndOfInput);
                }
                Some(c) => {
                    if is_whitespace(*c) {
                        observer.advance_one_char(*c);
                        s.next();
                    } else if *c == '}' {
                        let mut seen = BTreeSet::new();
                        for v in values_so_far.iter() {
                            if seen.contains(v) {
                                return Err(ParserError::DuplicateValueInSet { value: v.clone() });
                            }
                            seen.insert(v);
                        }
                        observer.stop_parsing_current();
                        observer.advance_one_char(*c);
                        s.next();
                        return Ok(Value::Set(values_so_far.iter().cloned().collect()));
                    } else {
                        let value = parse_helper(s, ParserState::Begin, observer, opts)?;
                        values_so_far.push(value);
                    }
                }
            },

            ParserState::ParsingAtom { ref mut built_up } => match s.peek() {
                None => {
                    return if built_up.is_empty() {
                        Err(ParserError::UnexpectedEndOfInput)
                    } else {
                        let value = interpret_atom(built_up)?;
                        observer.stop_parsing_current();
                        Ok(value)
                    }
                }
                Some(c) => {
                    if !is_allowed_atom_character(*c) {
                        let value = interpret_atom(built_up)?;
                        observer.stop_parsing_current();
                        return Ok(value);
                    } else {
                        built_up.push(*c);
                        observer.advance_one_char(*c);
                        s.next();
                    }
                }
            },

            ParserState::ParsingString { ref mut built_up } => match s.peek() {
                None => return Err(ParserError::UnexpectedEndOfInput),
                Some(c) => {
                    if *c == '"' {
                        observer.stop_parsing_current();
                        observer.advance_one_char(*c);
                        s.next();
                        return Ok(Value::String(built_up.clone()));
                    } else if *c == '\\' {
                        observer.advance_one_char(*c);
                        s.next();
                        match s.next() {
                            None => return Err(ParserError::InvalidStringEscape),
                            Some(c) => {
                                observer.advance_one_char(c);
                                match c {
                                    't' => built_up.push('\t'),
                                    'r' => built_up.push('\r'),
                                    'n' => built_up.push('\n'),

                                    '\\' => built_up.push('\\'),
                                    '"' => built_up.push('"'),
                                    'u' => match (s.next(), s.next(), s.next(), s.next()) {
                                        (Some(c1), Some(c2), Some(c3), Some(c4)) => {
                                            observer.advance_one_char(c1);
                                            observer.advance_one_char(c2);
                                            observer.advance_one_char(c3);
                                            observer.advance_one_char(c4);
                                            let str: String =
                                                [c1, c2, c3, c4].iter().copied().collect();
                                            let unicode = u32::from_str_radix(&str, 16)
                                                .map_err(|_| ParserError::InvalidStringEscape)?;
                                            match char::from_u32(unicode) {
                                                None => {
                                                    return Err(ParserError::InvalidStringEscape)
                                                }
                                                Some(c) => built_up.push(c),
                                            }
                                            continue 'parsing;
                                        }
                                        (Some(c1), Some(c2), Some(c3), _) => {
                                            observer.advance_one_char(c1);
                                            observer.advance_one_char(c2);
                                            observer.advance_one_char(c3);
                                            return Err(ParserError::InvalidStringEscape);
                                        }
                                        (Some(c1), Some(c2), _, _) => {
                                            observer.advance_one_char(c1);
                                            observer.advance_one_char(c2);
                                            return Err(ParserError::InvalidStringEscape);
                                        }
                                        (Some(c1), _, _, _) => {
                                            observer.advance_one_char(c1);
                                            return Err(ParserError::InvalidStringEscape);
                                        }
                                        _ => {
                                            return Err(ParserError::InvalidStringEscape);
                                        }
                                    },
                                    _ => return Err(ParserError::InvalidStringEscape),
                                }
                            }
                        }
                    } else {
                        built_up.push(*c);
                        observer.advance_one_char(*c);
                        s.next();
                    }
                }
            },
            ParserState::SelectingDispatch => {
                match s.peek() {
                    None => {
                        return Err(ParserError::UnexpectedEndOfInput);
                    }
                    Some(c) => {
                        if *c == '_' {
                            // Drop the next form. Still error if that form is malformed
                            observer.advance_one_char(*c);
                            s.next();
                            let _ = parse_helper(s, ParserState::Begin, observer, opts)?;
                        } else if *c == '{' {
                            observer.advance_one_char(*c);
                            s.next();
                            observer.start_parsing_set();
                            parser_state = ParserState::ParsingSet {
                                values_so_far: vec![],
                            };
                        } else {
                            // We expect to read a symbol next and we will associate that symbol as the tag of
                            // the following element
                            let value = parse_helper(s, ParserState::Begin, observer, opts)?;
                            match value {
                                Value::Symbol(symbol) => {
                                    let next_success =
                                        parse_helper(s, ParserState::Begin, observer, opts)?;

                                    // Handle builtin #inst
                                    if symbol.namespace == None && symbol.name == "inst" {
                                        if let Value::String(timestamp) = next_success {
                                            let datetime =
                                                chrono::DateTime::parse_from_rfc3339(&timestamp)
                                                    .map_err(|parse_error| {
                                                        ParserError::InvalidInst(Some(parse_error))
                                                    })?;
                                            return Ok(Value::Inst(datetime));
                                        } else {
                                            return Err(ParserError::InvalidInst(None));
                                        }
                                    }
                                    // Handle builtin #uuid
                                    else if symbol.namespace == None && symbol.name == "uuid" {
                                        if let Value::String(uuid_str) = next_success {
                                            let uuid = Uuid::parse_str(&uuid_str).map_err(
                                                |parse_error| {
                                                    ParserError::InvalidUuid(Some(parse_error))
                                                },
                                            )?;
                                            return Ok(Value::Uuid(uuid));
                                        } else {
                                            return Err(ParserError::InvalidUuid(None));
                                        }
                                    }
                                    // Everything else becomes a generic TaggedElement
                                    else {
                                        return Ok(Value::TaggedElement(
                                            symbol,
                                            Box::new(next_success),
                                        ));
                                    }
                                }
                                Value::Keyword(ref ns) => {
                                    if !opts.allow_namespaced_map_syntax || ns.namespace.is_some() {
                                        return Err(ParserError::InvalidElementForTag { value });
                                    } else {
                                        let next_success =
                                            parse_helper(s, ParserState::Begin, observer, opts)?;
                                        if let Value::Map(following_map) = next_success {
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
                                            return Ok(Value::Map(new_map));
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
                }
            }

            ParserState::ParsingCharacter => {
                let value = parse_helper(s, ParserState::Begin, observer, opts)?;
                return if let Value::Symbol(symbol) = value {
                    if symbol.namespace == None {
                        if symbol.name.len() == 1 {
                            Ok(Value::Character(symbol.name.chars().next().expect(
                                "Asserted that this string has at least one character.",
                            )))
                        } else {
                            match symbol.name.as_str() {
                                "newline" => Ok(Value::Character('\n')),
                                "return" => Ok(Value::Character('\r')),
                                "space" => Ok(Value::Character(' ')),
                                "tab" => Ok(Value::Character('\t')),
                                _ => Err(ParserError::InvalidCharacterSpecification),
                            }
                        }
                    } else {
                        Err(ParserError::InvalidCharacterSpecification)
                    }
                } else {
                    Err(ParserError::InvalidCharacterSpecification)
                };
            }
        }
    }
}

/// Options you can pass to the EDN parser.
#[derive(Debug, Copy, Clone)]
pub struct ParserOptions {
    /// Whether to allow the #some.ns{:key "val"} syntax that was introduced in clojure 1.9
    /// but not reflected in the EDN spec.
    ///
    /// Defaults to true.
    pub allow_namespaced_map_syntax: bool,
}

impl Default for ParserOptions {
    fn default() -> Self {
        ParserOptions {
            allow_namespaced_map_syntax: true,
        }
    }
}

/// Parse EDN from the given input string with the given options.
pub fn parse_str_with_options(
    s: &str,
    opts: ParserOptions,
) -> Result<Value, ParserErrorWithContext> {
    let mut chars = s.chars().peekable();
    let mut context = ContextStackerObserver::new();
    let value =
        parse_helper(&mut chars, ParserState::Begin, &mut context, &opts).map_err(|err| {
            ParserErrorWithContext {
                context: context.context.clone(),
                row: context.row_col.row,
                col: context.row_col.col,
                error: err,
            }
        })?;
    Ok(value)
}

/// Parse EDN from the given input string with default options.
pub fn parse_str(s: &str) -> Result<Value, ParserErrorWithContext> {
    parse_str_with_options(s, ParserOptions::default())
}

impl FromStr for Value {
    type Err = ParserErrorWithContext;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_str(s)
    }
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

/// Emit the given EDN value as a String.
pub fn emit_str(value: &Value) -> String {
    format!("{}", value)
}

/// Parser which can yield multiple forms from a single iterator of
/// chars
pub struct Parser<Iter: Iterator<Item = char>> {
    opts: ParserOptions,
    iter: Peekable<Iter>,
}

impl<'a> Parser<Chars<'a>> {
    /// Construct a parser from a &str,
    pub fn from_str(s: &'a str, opts: ParserOptions) -> Parser<Chars<'a>> {
        Parser {
            opts,
            iter: s.chars().peekable(),
        }
    }
}

impl<Iter: Iterator<Item = char>> Parser<Iter> {
    /// Construct a parser from an arbitrary iterator,
    pub fn from_iter(iter: Iter, opts: ParserOptions) -> Parser<Iter> {
        Parser {
            opts,
            iter: iter.peekable(),
        }
    }
}

impl<Iter: Iterator<Item = char>> Iterator for Parser<Iter> {
    type Item = Result<Value, ParserError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut context = ContextStackerObserver::new();
        match parse_helper(&mut self.iter, ParserState::Begin, &mut context, &self.opts) {
            Err(error) => {
                if error == ParserError::EmptyInput {
                    None
                } else {
                    Some(Err(error))
                }
            }
            Ok(value) => Some(Ok(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::DateTime;
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
    fn test_access_symbol_parts() {
        let sym = Symbol::from_name("abc");
        assert_eq!((None, "abc"), (sym.namespace(), sym.name()));

        let sym = Symbol::from_namespace_and_name("abc", "def");
        assert_eq!((Some("abc"), "def"), (sym.namespace(), sym.name()));
    }

    #[test]
    fn test_access_keyword_parts() {
        let kw = Keyword::from_name("abc");
        assert_eq!((None, "abc"), (kw.namespace(), kw.name()));

        let kw = Keyword::from_namespace_and_name("abc", "def");
        assert_eq!((Some("abc"), "def"), (kw.namespace(), kw.name()));
    }

    #[test]
    fn test_parsing_empty_list() {
        assert_eq!(Value::List(vec![]), parse_str("()").unwrap())
    }

    #[test]
    fn test_parsing_empty_vector() {
        assert_eq!(Value::Vector(vec![]), parse_str("[]").unwrap())
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
            parse_str("[()[]()[]]").unwrap()
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
            parse_str("   ,, , [ ,, , ,()[,,,]( ) []]").unwrap()
        )
    }

    #[test]
    fn test_parsing_empty_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![])),
            parse_str("{}").unwrap()
        )
    }

    #[test]
    fn test_parsing_uneven_map() {
        assert_eq!(
            Err(ParserError::OddNumberOfMapElements),
            parse_str("{()}").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::OddNumberOfMapElements),
            parse_str("{() [] []}").map_err(|err| err.error)
        )
    }

    #[test]
    fn test_parsing_even_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::List(vec![]),
                Value::List(vec![])
            )])),
            parse_str("{() ()}").unwrap()
        );
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![
                (Value::List(vec![]), Value::Vector(vec![])),
                (Value::Keyword(Keyword::from_name("a")), Value::List(vec![]))
            ])),
            parse_str("{()[] :a ()}").unwrap()
        )
    }

    #[test]
    fn test_parsing_duplicate_map_keys() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::List(vec![]),
                Value::List(vec![])
            )])),
            parse_str("{() ()}").unwrap()
        );
        assert_eq!(
            Err(ParserError::DuplicateKeyInMap {
                value: Value::List(vec![])
            }),
            parse_str("{()[] () ()}").map_err(|err| err.error)
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
            parse_str("\"\\uAAAA\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_string_nested() {
        assert_eq!(
            Value::Vector(vec![Value::String("ꪪ".to_string())]),
            parse_str("[\"\\uAAAA\"]").unwrap()
        )
    }

    #[test]
    fn test_parsing_multiline_string() {
        assert_eq!(
            Value::String("abc\n    \ndef    \n".to_string()),
            parse_str("\"abc\n    \ndef    \n\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_string_map() {
        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::String("abc".to_string()),
                Value::String("def".to_string())
            )])),
            parse_str("{\"abc\" \"def\"}").unwrap()
        );

        assert_eq!(
            Value::Map(BTreeMap::from_iter(vec![(
                Value::String("abc".to_string()),
                Value::String("def".to_string())
            )])),
            parse_str("{\"abc\"\"def\"}").unwrap()
        )
    }

    #[test]
    fn test_parsing_inst() {
        assert_eq!(
            Value::Inst(DateTime::parse_from_rfc3339("1985-04-12T23:20:50.52Z").unwrap()),
            parse_str("#inst\"1985-04-12T23:20:50.52Z\"").unwrap()
        )
    }

    #[test]
    fn test_parsing_uuid() {
        assert_eq!(
            Value::Uuid(Uuid::parse_str("f81d4fae-7dec-11d0-a765-00a0c91e6bf6").unwrap()),
            parse_str("#uuid\"f81d4fae-7dec-11d0-a765-00a0c91e6bf6\"").unwrap()
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
            parse_str("[ a  abc abc/def -> / my.org/stuff ]").unwrap()
        );
    }

    #[test]
    fn test_parsing_symbol_errs() {
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtBeginningOfSymbol),
            parse_str("/abc").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfSymbol),
            parse_str("abc/").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfSymbol),
            parse_str("abc/ ").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfSymbol),
            parse_str("abc/ []").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveMoreThanOneSlashInSymbol),
            parse_str("a/b/c").map_err(|err| err.error)
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
            parse_str("[ :a  :abc :abc/def :-> :/ :my.org/stuff ]").unwrap()
        );
    }

    #[test]
    fn test_parsing_keyword_errs() {
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtBeginningOfKeyword),
            parse_str(":/abc").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfKeyword),
            parse_str(":abc/").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfKeyword),
            parse_str(":abc/ ").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveSlashAtEndOfKeyword),
            parse_str(":abc/ []").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveMoreThanOneSlashInKeyword),
            parse_str(":a/b/c").map_err(|err| err.error)
        );
        assert_eq!(
            Err(ParserError::CannotHaveColonInKeyword),
            parse_str("::namespaced").map_err(|err| err.error)
        );
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
            parse_str("#{:abc :def :ghi/jkl }").unwrap()
        );

        assert_eq!(
            Err(ParserError::DuplicateValueInSet {
                value: Value::Symbol(Symbol::from_name("a"))
            }),
            parse_str("#{a b c d e f a g h}").map_err(|err| err.error)
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
            parse_str(
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
        assert!(equal(
            &parse_str("name").unwrap(),
            &parse_str("name").unwrap()
        ));
        assert!(equal(
            &parse_str("person/name").unwrap(),
            &parse_str("person/name").unwrap()
        ));
        assert!(equal(
            &parse_str(":name").unwrap(),
            &parse_str(":name").unwrap()
        ));
        assert!(equal(
            &parse_str(":person/name").unwrap(),
            &parse_str(":person/name").unwrap()
        ));

        // Had an issue with whitespace
        assert!(equal(
            &parse_str("name ").unwrap(),
            &parse_str("name ").unwrap()
        ));
        assert!(equal(
            &parse_str("person/name ").unwrap(),
            &parse_str("person/name ").unwrap()
        ));
        assert!(equal(
            &parse_str(":name ").unwrap(),
            &parse_str(":name ").unwrap()
        ));
        assert!(equal(
            &parse_str(":person/name ").unwrap(),
            &parse_str(":person/name ").unwrap()
        ));
    }

    #[test]
    fn test_complex_equals() {
        assert!(equal(
            &parse_str(
                "\
            {:person/parent \"Bob\"\
             :person/name \"Joe\"\
             :ssn \"123\"\
             friends [\"sally\" \"john\" \"linda\"]\
             \"other\" {:stuff :here}}"
            )
            .unwrap(),
            &parse_str(
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
            parse_str("(nil false true)").unwrap()
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
            parse_str("{\\space \\z\\a \\newline \\b \\return \\r \\c \\tab \\d}").unwrap()
        )
    }

    #[test]
    fn test_parse_int() {
        assert_eq!(Value::Integer(123), parse_str("123").unwrap())
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(Value::Float(OrderedFloat(12.1)), parse_str("12.1").unwrap())
    }

    #[test]
    fn test_parse_neg_int() {
        assert_eq!(Value::Integer(-123), parse_str("-123").unwrap())
    }

    #[test]
    fn test_parse_neg_float() {
        assert_eq!(
            Value::Float(OrderedFloat(-12.1)),
            parse_str("-12.1").unwrap()
        )
    }

    #[test]
    fn test_parse_pos_int() {
        assert_eq!(Value::Integer(123), parse_str("+123").unwrap())
    }

    #[test]
    fn test_parse_pos_float() {
        assert_eq!(
            Value::Float(OrderedFloat(12.1)),
            parse_str("+12.1").unwrap()
        )
    }

    #[test]
    fn test_parse_zero() {
        assert_eq!(Value::Integer(0), parse_str("+0").unwrap(),);
        assert_eq!(Value::Integer(0), parse_str("0").unwrap(),);
        assert_eq!(Value::Integer(0), parse_str("-0").unwrap(),);
    }

    #[test]
    fn test_parse_zero_float() {
        assert_eq!(Value::Float(OrderedFloat(0f64)), parse_str("+0.").unwrap());
        assert_eq!(Value::Float(OrderedFloat(0f64)), parse_str("0.").unwrap());
        assert_eq!(Value::Float(OrderedFloat(0f64)), parse_str("-0.").unwrap());
    }

    #[test]
    fn test_parse_e() {
        assert_eq!(
            Value::Float(OrderedFloat(1000.0)),
            parse_str("10e+2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(1200.0)),
            parse_str("12e+2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(1200.0)),
            parse_str("12e2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(1200.0)),
            parse_str("12E2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(5200.0)),
            parse_str("52E+2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(1.2)),
            parse_str("120e-2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(1.2)),
            parse_str("120E-2").unwrap()
        );
        assert_eq!(
            Value::Float(OrderedFloat(1422141241242142142141241.124)),
            parse_str("1422141241242142142141241.124E0").unwrap()
        );
    }

    #[test]
    fn test_parse_bigint() {
        assert_eq!(Value::BigInt(BigInt::from(123)), parse_str("123N").unwrap());
    }

    #[test]
    fn test_parse_bigdec() {
        assert_eq!(
            Value::BigDec(BigDecimal::from(123)),
            parse_str("123M").unwrap()
        );
    }

    #[test]
    fn test_bad_bigdec() {
        assert!(match parse_str("12a3M").map_err(|err| err.error) {
            Err(ParserError::BadBigDec { .. }) => true,
            _ => false,
        });
    }

    #[test]
    fn test_bad_bigint() {
        assert!(match parse_str("12a3N").map_err(|err| err.error) {
            Err(ParserError::BadBigInt { .. }) => true,
            _ => false,
        });
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
            parse_str("( 1 2 3 \"abc\"\n;; so here is where we do some wacky stuff \n [a b/qq ; and then here\n c \n \n;; aaa\n{12 34.5 :a \n;;aaadeafaef\nb}])").unwrap()
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
        assert_eq!(emit_str(&v1), emit_str(&parse_str(&emit_str(&v1)).unwrap()));
        assert_eq!(emit_str(&v2), emit_str(&parse_str(&emit_str(&v2)).unwrap()));

        assert_eq!(
            parse_str(&emit_str(&v1)).unwrap(),
            parse_str(&emit_str(&parse_str(&emit_str(&v1)).unwrap())).unwrap()
        );
        assert_eq!(
            parse_str(&emit_str(&v2)).unwrap(),
            parse_str(&emit_str(&parse_str(&emit_str(&v2)).unwrap())).unwrap()
        );
    }

    #[test]
    fn test_big_vector() {
        let mut vals: Vec<Value> = vec![];
        for i in 0..100000 {
            vals.push(Value::Integer(i))
        }
        let ser = emit_str(&Value::Vector(vals.clone()));
        // shouldn't stack overflow
        assert_eq!(parse_str(&ser).unwrap(), Value::Vector(vals));
    }

    #[test]
    fn test_big_list() {
        let mut vals: Vec<Value> = vec![];
        for i in 0..100000 {
            vals.push(Value::Integer(i))
        }
        let ser = emit_str(&Value::List(vals.clone()));
        assert_eq!(parse_str(&ser).unwrap(), Value::List(vals));
    }

    #[test]
    fn test_big_set() {
        let mut vals: Vec<Value> = vec![];
        for i in 0..100000 {
            vals.push(Value::Integer(i))
        }
        let ser = emit_str(&Value::Set(BTreeSet::from_iter(vals.clone())));
        assert_eq!(
            parse_str(&ser).unwrap(),
            Value::Set(BTreeSet::from_iter(vals))
        );
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
        let ser = emit_str(&Value::Map(BTreeMap::from_iter(vals.clone())));
        assert_eq!(
            parse_str(&ser).unwrap(),
            Value::Map(BTreeMap::from_iter(vals))
        );
    }

    #[test]
    fn test_two_colons() {
        assert_eq!(
            Err(ParserError::CannotHaveColonInKeyword),
            parse_str("::").map_err(|err| err.error)
        )
    }

    #[test]
    fn test_row_col_tracking() {
        assert_eq!(
            Err(ParserErrorWithContext {
                context: vec![Context::ParsingAtom { row: 5, col: 1 }],
                row: 6,
                col: 1,
                error: ParserError::CannotHaveColonInKeyword
            }),
            parse_str("    ::",)
        );
        assert_eq!(
            Err(ParserErrorWithContext {
                context: Vec::new(),
                row: 0,
                col: 1,
                error: ParserError::EmptyInput
            }),
            parse_str("",)
        );
        assert_eq!(
            Err(ParserErrorWithContext {
                context: vec![Context::ParsingAtom { row: 4, col: 1 }],
                row: 5,
                col: 1,
                error: ParserError::CannotHaveColonInKeyword
            }),
            parse_str("   ::",)
        );
        assert_eq!(
            Err(ParserErrorWithContext {
                context: vec![Context::ParsingAtom { row: 1, col: 3 }],
                row: 2,
                col: 3,
                error: ParserError::CannotHaveColonInKeyword
            }),
            parse_str("   \n\n::",)
        );
    }

    #[test]
    fn test_context_maintaining() {
        assert_eq!(
            Err(ParserErrorWithContext {
                context: vec![
                    Context::ParsingVector { row: 2, col: 1 },
                    Context::ParsingAtom { row: 8, col: 1 }
                ],
                row: 10,
                col: 1,
                error: ParserError::CannotHaveColonInKeyword
            }),
            parse_str(" [ 1 2 ::a]",)
        );

        assert_eq!(
            Err(ParserErrorWithContext {
                context: vec![
                    Context::ParsingList { row: 2, col: 1 },
                    Context::ParsingSet { row: 2, col: 2 },
                    Context::ParsingMap { row: 1, col: 3 },
                    Context::ParsingVector { row: 3, col: 3 },
                    Context::ParsingAtom { row: 3, col: 5 }
                ],
                row: 5,
                col: 5,
                error: ParserError::CannotHaveColonInKeyword
            }),
            parse_str(" ( a b c \n#{ \n{ [ \n1 2 4\n  ::a  \n3]  3} } )",)
        );
        assert_eq!(
            Err(ParserErrorWithContext {
                context: vec![Context::ParsingList { row: 2, col: 1 }],
                row: 8,
                col: 1,
                error: ParserError::UnexpectedEndOfInput
            }),
            parse_str(" ( [] {}",)
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
            parse_str_with_options(
                "#:apple.tree{:c 1 \"a\" 2 :d 3 :d.e/f :a}",
                ParserOptions {
                    allow_namespaced_map_syntax: true,
                    ..ParserOptions::default()
                }
            )
            .unwrap()
        )
    }

    #[test]
    fn test_parsing_symbols_that_might_be_mistaken_for_numbers() {
        assert_eq!(
            parse_str("a1b").unwrap(),
            Value::Symbol(Symbol::from_name("a1b"))
        );
        assert_eq!(
            parse_str("ab1").unwrap(),
            Value::Symbol(Symbol::from_name("ab1"))
        );
        assert_eq!(
            parse_str("ab1/c1d").unwrap(),
            Value::Symbol(Symbol::from_namespace_and_name("ab1", "c1d"))
        );
        assert_eq!(
            parse_str("ab1/cd1").unwrap(),
            Value::Symbol(Symbol::from_namespace_and_name("ab1", "cd1"))
        )
    }

    #[test]
    fn test_bad_string_escape() {
        assert_eq!(
            parse_str("\"\\u\"").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
        assert_eq!(
            parse_str("\"\\u1\"").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
        assert_eq!(
            parse_str("\"\\u12\"").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
        assert_eq!(
            parse_str("\"\\u123\"").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );

        assert_eq!(
            parse_str("\"\\u").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
        assert_eq!(
            parse_str("\"\\u1").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
        assert_eq!(
            parse_str("\"\\u12").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
        assert_eq!(
            parse_str("\"\\u123").map_err(|err| err.error),
            Err(ParserError::InvalidStringEscape)
        );
    }

    // Tests here taken from
    // https://github.com/utkarshkukreti/edn.rs/blob/master/tests/from_tests.rs#L9
    #[test]
    fn from_bool() {
        assert_eq!(Value::from(true), Value::Boolean(true));
        assert_eq!(Value::from(false), Value::Boolean(false));
    }
    #[test]
    fn from_str() {
        assert_eq!(Value::from(""), Value::String("".to_string()));
        assert_eq!(Value::from("hello"), Value::String("hello".to_string()));
    }

    #[test]
    fn from_string() {
        assert_eq!(Value::from("".to_string()), Value::String("".to_string()));
        assert_eq!(
            Value::from("hello".to_string()),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn from_char() {
        assert_eq!(Value::from('c'), Value::Character('c'));
    }

    #[test]
    fn from_num() {
        assert_eq!(Value::from(0_i64), Value::Integer(0));
        assert_eq!(Value::from(0), Value::Integer(0));
        assert_eq!(Value::from(-1), Value::Integer(-1));

        assert_eq!(Value::from(0_f64), Value::Float(OrderedFloat(0_f64)));
        assert_eq!(Value::from(0_f64), Value::Float(OrderedFloat(0_f64)));

        assert_eq!(Value::from(0_f64), Value::Float(OrderedFloat(0_f64)));
        assert_eq!(
            Value::from(OrderedFloat(0_f64)),
            Value::Float(OrderedFloat(0_f64))
        );
    }
    // -------------------------------------------------------------------------

    #[test]
    fn from_unit() {
        assert_eq!(Value::Nil, Value::from(()))
    }

    #[test]
    fn from_symbol() {
        assert_eq!(
            Value::Symbol(Symbol::from_name("abc")),
            Value::from(Symbol::from_name("abc"))
        );

        assert_eq!(
            Value::Symbol(Symbol::from_namespace_and_name("abc", "def")),
            Value::from(Symbol::from_namespace_and_name("abc", "def"))
        );
    }

    #[test]
    fn from_keyword() {
        assert_eq!(
            Value::Keyword(Keyword::from_name("abc")),
            Value::from(Keyword::from_name("abc"))
        );

        assert_eq!(
            Value::Keyword(Keyword::from_namespace_and_name("abc", "def")),
            Value::from(Keyword::from_namespace_and_name("abc", "def"))
        );
    }

    #[test]
    fn from_bigint() {
        assert_eq!(
            Value::BigInt(BigInt::from(123)),
            Value::from(BigInt::from(123))
        );
    }

    #[test]
    fn from_bigdec() {
        assert_eq!(
            Value::BigDec(BigDecimal::from(123)),
            Value::from(BigDecimal::from(123))
        );
    }

    #[test]
    fn from_datetime() {
        assert_eq!(
            Value::Inst(DateTime::parse_from_rfc3339("1985-04-12T23:20:50.52Z").unwrap()),
            Value::from(DateTime::parse_from_rfc3339("1985-04-12T23:20:50.52Z").unwrap())
        )
    }

    #[test]
    fn from_uuid() {
        assert_eq!(
            Value::Uuid(Uuid::parse_str("f81d4fae-7dec-11d0-a765-00a0c91e6bf6").unwrap()),
            Value::from(Uuid::parse_str("f81d4fae-7dec-11d0-a765-00a0c91e6bf6").unwrap())
        )
    }

    #[test]
    fn test_symbols_with_colons() {
        assert_eq!(
            parse_str("a:").map_err(|err| err.error),
            Err(ParserError::CannotHaveColonInSymbol)
        );
        assert_eq!(
            parse_str("a:b").map_err(|err| err.error),
            Err(ParserError::CannotHaveColonInSymbol)
        );
        assert_eq!(
            parse_str("a:b/c").map_err(|err| err.error),
            Err(ParserError::CannotHaveColonInSymbol)
        );
        assert_eq!(
            parse_str("ab/c:").map_err(|err| err.error),
            Err(ParserError::CannotHaveColonInSymbol)
        );
        assert_eq!(
            parse_str("ab/c:d").map_err(|err| err.error),
            Err(ParserError::CannotHaveColonInSymbol)
        );
        assert_eq!(
            parse_str("ab/c: ").map_err(|err| err.error),
            Err(ParserError::CannotHaveColonInSymbol)
        );
    }

    #[test]
    fn test_many_values() {
        let mut parser = Parser::from_str("123 456 [] [[]]", ParserOptions::default());
        assert_eq!(
            (
                Some(Ok(Value::from(123))),
                Some(Ok(Value::from(456))),
                Some(Ok(Value::Vector(vec![]))),
                Some(Ok(Value::Vector(vec![Value::Vector(vec![])]))),
                None,
                None
            ),
            (
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next()
            )
        )
    }

    #[test]
    fn test_many_values_with_errors() {
        let mut parser = Parser::from_str("123 456 :: [] :: [[]]", ParserOptions::default());
        assert_eq!(
            (
                Some(Ok(Value::from(123))),
                Some(Ok(Value::from(456))),
                Some(Err(ParserError::CannotHaveColonInKeyword)),
                Some(Ok(Value::Vector(vec![]))),
                Some(Err(ParserError::CannotHaveColonInKeyword)),
                Some(Ok(Value::Vector(vec![Value::Vector(vec![])]))),
                None,
                None
            ),
            (
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next()
            )
        );
    }

    #[test]
    fn test_many_values_with_errors_iter() {
        let mut parser =
            Parser::from_iter("123 456 :: [] :: [[]]".chars(), ParserOptions::default());
        assert_eq!(
            (
                Some(Ok(Value::from(123))),
                Some(Ok(Value::from(456))),
                Some(Err(ParserError::CannotHaveColonInKeyword)),
                Some(Ok(Value::Vector(vec![]))),
                Some(Err(ParserError::CannotHaveColonInKeyword)),
                Some(Ok(Value::Vector(vec![Value::Vector(vec![])]))),
                None,
                None
            ),
            (
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next(),
                parser.next()
            )
        );
    }
}
