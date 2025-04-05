// src/util.rs
use std::collections::HashMap;
use crate::parser::{Action, ParseTable, Rule};

pub fn load_parse_table(
    rules: &HashMap<usize, Rule>,
    states_dict: HashMap<String, HashMap<String, (String, String)>>, 
    start: &str
) -> ParseTable<usize> {
    let mut table = ParseTable::<usize>::new();
    
    // Setup start and end states based on the start symbol
    let start_state = 9;  // Use state 9 for start by default
    let end_state = 27;   // Use state 27 for end by default
    
    table.start_states.insert(start.to_string(), start_state);
    table.end_states.insert(start.to_string(), end_state);
    
    // Convert serialized states to a ParseTable
    for (state_str, transitions) in states_dict {
        let state = state_str.parse::<usize>().unwrap_or(0);
        let mut state_transitions = HashMap::new();
        
        for (symbol, (action_type, action_value)) in transitions {
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
                },
                "accept" => Action::Accept,
                _ => Action::Error,
            };
            
            state_transitions.insert(symbol.clone(), action);
        }
        
        table.states.insert(state, state_transitions);
    }
    
    table
}