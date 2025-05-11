// src/util.rs
use crate::parser::{Action, ParseTable, Rule};
use std::collections::HashMap;

pub fn load_parse_table(
    rules: &HashMap<usize, Rule>,
    states_dict: HashMap<String, HashMap<String, (String, String)>>,
    start: &str,
    start_state: usize,
    end_state: usize,
) -> ParseTable<usize> {
    let mut table = ParseTable::<usize>::new();

    table.start_states.insert(start.to_string(), start_state);
    table.end_states.insert(start.to_string(), end_state);

    // Convert serialized states to a ParseTable
    for (state_str, transitions) in states_dict {
        let state = state_str.parse::<usize>().unwrap_or(0);
        let mut state_transitions = HashMap::new();

        for (symbol, (action_type, action_value)) in transitions {
            // eprintln!("Action: '{}' -> '{}'", action_type, action_value);

            let action = match action_type.as_str() {
                "shift" => Action::Shift(action_value.parse::<usize>().unwrap_or(0)),
                "reduce" => {
                    let rule_id = action_value.parse::<usize>().unwrap_or(0);
                    if let Some(rule) = rules.get(&rule_id) {
                        Action::Reduce(rule.clone())
                    } else {
                        // Default to an empty rule if not found
                        Action::Reduce(Rule::new(rule_id, "unknown".to_string(), vec![]))
                    }
                }
                // "accept" => Action::Accept,
                _ => Action::Error,
            };

            state_transitions.insert(symbol.clone(), action);
        }

        table.states.insert(state, state_transitions);
    }

    table
}
