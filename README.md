[![Run Tests](https://github.com/bowbahdoe/edn_format/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/bowbahdoe/edn_format/actions/workflows/run_tests.yml)

# edn_format

This crate provides an implementation of the [EDN format](https://github.com/edn-format/edn) for rust.

The intent is to provide a more complete api than the existing [edn](https://crates.io/crates/edn) and
[edn-rs](https://crates.io/crates/edn-rs) crates.

```
[dependencies]
edn-format = "1.1.1"
```

## Example usage
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