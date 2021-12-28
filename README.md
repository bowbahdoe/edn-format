[![Run Tests](https://github.com/bowbahdoe/edn_format/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/bowbahdoe/edn_format/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/bowbahdoe/edn-format/branch/main/graph/badge.svg?token=4YL2AFOIUE)](https://codecov.io/gh/bowbahdoe/edn-format)
# edn-format

This crate provides an implementation of the [EDN format](https://github.com/edn-format/edn) for rust.

The intent is to provide a more complete api than the existing [edn](https://crates.io/crates/edn) and
[edn-rs](https://crates.io/crates/edn-rs) crates.

```
[dependencies]
edn-format = "3.0.1"
```

## Example usage

### Round trip data
```rust
let data = "{:person/name    \"bob\"\
             :person/age      35\
             :person/children #{\"sally\" \"suzie\" \"jen\"}}";
let parsed = parse_str(data).expect("Should be valid");

println!("{:?}", parsed);
// Map({Keyword(Keyword { namespace: Some("person"), name: "age" }): Integer(35), Keyword(Keyword { namespace: Some("person"), name: "name" }): String("bob"), Keyword(Keyword { namespace: Some("person"), name: "children" }): Set({String("jen"), String("sally"), String("suzie")})})

println!("{}", emit_str(&parsed));
// {:person/age 35 :person/name "bob" :person/children #{"jen" "sally" "suzie"}}
```

### Round trip from user defined struct
You will likely notice that writing code to serialize and deserialize your own data structures
using just the facilities in this library will lead to some verbose code. 

EDN's semantics are much richer than JSON's and providing something like serde support is a problem I deliberately 
chose not to solve.

There are pros and cons to this, but I choose to focus on the pro that you will at least have explicit
control over the form your serialized structures take.

```rust 
#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
    hobbies: Vec<String>,
}

impl Into<Value> for Person {
    fn into(self) -> Value {
        Value::TaggedElement(
            Symbol::from_namespace_and_name("my.project", "person"),
            Box::new(Value::Map(BTreeMap::from([
                (
                    Value::from(Keyword::from_name("name")),
                    Value::from(self.name),
                ),
                (
                    Value::from(Keyword::from_name("age")),
                    Value::from(self.age as i64),
                ),
                (
                    Value::from(Keyword::from_name("hobbies")),
                    Value::Vector(
                        self.hobbies
                            .into_iter()
                            .map(|hobby| Value::from(hobby))
                            .collect(),
                    ),
                ),
            ]))),
        )
    }
}

impl TryFrom<Value> for Person {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::TaggedElement(tag, element) => {
                if tag == Symbol::from_namespace_and_name("my.project", "person") {
                    match *element {
                        Value::Map(map) => {
                            if let (
                                Some(Value::String(name)),
                                Some(Value::Integer(age)),
                                Some(Value::Vector(hobbies)),
                            ) = (
                                map.get(&Value::from(Keyword::from_name("name"))),
                                map.get(&Value::from(Keyword::from_name("age"))),
                                map.get(&Value::from(Keyword::from_name("hobbies"))),
                            ) {
                                let mut hobby_strings = vec![];
                                for hobby in hobbies {
                                    if let Value::String(hobby) = hobby {
                                        hobby_strings.push(hobby.clone())
                                    }
                                    else {
                                        return Err(())
                                    }
                                }
                                Ok(Person {
                                    name: name.clone(),
                                    age: *age as u32,
                                    hobbies: hobby_strings
                                })
                            } else {
                                Err(())
                            }
                        }
                        _ => Err(()),
                    }
                } else {
                    Err(())
                }
            }
            // I'm sure this error handling strategy isn't going
            // to win many awards
            _ => Err(()),
        }
    }
}


fn example() {
    let bob = Person {
        name: "bob".to_string(),
        age: 23,
        hobbies: vec!["card games".to_string(), "motorcycles".to_string()],
    };

    let serialized = emit_str(&bob.into());
    println!("{}", serialized);
    // #my.project/person {:age 23 :name "bob" :hobbies ["card games" "motorcycles"]}
    let deserialized = parse_str(&serialized).map(|value| Person::try_from(value));
    println!("{:?}", deserialized)
    // Ok(Ok(Person { name: "bob", age: 23, hobbies: ["card games", "motorcycles"] }))
}
```

### Parse iterator of data
```rust 
use edn_format::{Parser, ParserOptions};

let parser = Parser::from_iter("123 456 [] [[]]".chars(), ParserOptions::default());
for element in parser {
    println!("{}", element.expect("expected valid element"));
}

// 123
// 456
// []
// [[]]
```