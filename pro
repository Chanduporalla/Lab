exp - 1 :
def clean_c_file(input_file_path, output_file_path):
with open(input_file_path, 'r') as infile:
lines = infile.readlines()
cleaned_lines = []
in_multiline_comment = False
for line in lines:
stripped_line = line.rstrip()
if not stripped_line:
continue
if '/*' in stripped_line:
in_multiline_comment = True
if in_multiline_comment:
if '*/' in stripped_line:
in_multiline_comment = False
continue
line_without_comment = re.sub(r'//.*', '', stripped_line)
indentation = len(line) - len(line.lstrip())
line_without_extra_spaces = re.sub(r'\s+', ' ', line_without_comment).strip()
if line_without_extra_spaces:
cleaned_lines.append(' ' * indentation + line_without_extra_spaces)
with open(output_file_path, 'w') as outfile:
outfile.write('\n'.join(cleaned_lines))
input_file_name = next(iter(uploaded))
output_file_name = 'Updated_' + input_file_name
clean_c_file(input_file_name, output_file_name)
files.download(output_file_name)
print(f"Cleaned C file has been saved to: {output_file_name}")
exp - 2 :
import json
from collections import defaultdict
def read_fa():
# Input for states
states = input("Enter states (comma-separated): ").split(',')
states = [state.strip() for state in states]
alphabet = input("Enter alphabet (comma-separated): ").split(',')
alphabet = [symbol.strip() for symbol in alphabet]
transitions = {}
print("Enter transitions in format 'state:input => next_state' (type 'done' to finish):")
while True:
line = input().strip()
if line.lower() == "done":
break
if '=>' not in line:
print("Invalid format. Use 'state:input => next_state'.")
continue
try:
state_input, next_state = line.split('=>')
state, input_symbol = state_input.split(':')
state, input_symbol, next_state = state.strip(), input_symbol.strip(),
next_state.strip()
transitions.setdefault(state, {}).setdefault(input_symbol, set()).add(next_state)
except ValueError:
print("Invalid format. Use 'state:input => next_state'.")
start_state = input("Enter start state: ").strip()
accepting_states = input("Enter accepting states (comma-separated): ").split(',')
accepting_states = {state.strip() for state in accepting_states}
return {
"states": states,
"alphabet": alphabet,
"transitions": transitions,
"start_state": start_state,
"accepting_states": accepting_states
}
def epsilon_closure(nfa, start):
closure, stack = set(), [start]
while stack:
state = stack.pop()
if state not in closure:
closure.add(state)
stack.extend(nfa.get(state, {}).get('epsilon', []))
return closure
if __name__ == "__main__":
fa = read_fa()
epsilon_nfa = fa["transitions"]
start_state = fa["start_state"]
print("\nEpsilon Closures:")
closures = {state: epsilon_closure(epsilon_nfa, state) for state in fa["states"]}
for state, closure in closures.items():
print(f"Epsilon closure of {state}: {closure}")
input :
Enter states (comma-separated): 0,1,2,3
Enter alphabet (comma-separated): a,b
Enter transitions in format 'state:input => next_state' (type 'done' to finish):
0:epsilon = > 1
Invalid format. Use 'state:input => next_state'.
0:epsilon => 1
1:epsilon => 2
2:a=> 3
done
Enter start state: 0
Enter accepting states (comma-separated): 3
Epsilon Closures:
Epsilon closure of 0: {'1', '0', '2'}
Epsilon closure of 1: {'1', '2'}
Epsilon closure of 2: {'2'}
Epsilon closure of 3: {'3'}
exp - 3:
from collections import defaultdict
def read_fa():
states = input("Enter states (comma-separated): ").split(',')
states = [state.strip() for state in states]
alphabet = input("Enter alphabet (comma-separated): ").split(',')
alphabet = [symbol.strip() for symbol in alphabet]
transitions = defaultdict(lambda: defaultdict(set))
print("Enter transitions in format 'state:input => next_state' (type 'done' to finish):")
while True:
line = input().strip()
if line.lower() == "done":
break
if '=>' not in line or ':' not in line:
print("Invalid format. Use 'state:input => next_state'.")
continue
try:
state_input, next_state = line.split('=>')
state, input_symbol = state_input.split(':')
state, input_symbol, next_state = state.strip(), input_symbol.strip(),
next_state.strip()
transitions[state][input_symbol].add(next_state)
except ValueError:
print("Invalid format. Use 'state:input => next_state'.")
start_state = input("Enter start state: ").strip()
accepting_states = set(input("Enter accepting states (comma-separated): ").split(','))
return {
"states": states,
"alphabet": alphabet,
"transitions": transitions,
"start_state": start_state,
"accepting_states": {state.strip() for state in accepting_states}
}
def epsilon_closure(nfa, state):
closure, stack = {state}, [state]
while stack:
current = stack.pop()
for nxt in nfa.get(current, {}).get('epsilon', []):
if nxt not in closure:
closure.add(nxt)
stack.append(nxt)
return closure
def remove_epsilon_transitions(nfa, finals):
new_nfa, new_finals = defaultdict(dict), set()
closures = {s: epsilon_closure(nfa, s) for s in nfa}
for state, closure in closures.items():
for c in closure:
for symbol, targets in nfa.get(c, {}).items():
if symbol != 'epsilon':
new_nfa[state].setdefault(symbol, set()).update(targets)
if closure & finals:
new_finals.update(closure) # Fix: Add all closure states, not just `state`
return new_nfa, new_finals
if __name__ == "__main__":
nfa_data = read_fa()
nfa, new_finals = remove_epsilon_transitions(nfa_data["transitions"],
nfa_data["accepting_states"])
print("\nNFA without epsilon transitions:")
for state, trans in nfa.items():
for sym, targets in trans.items():
print(f"{state} --{sym}--> {targets}")
print("Final states:", new_finals)
input:
Enter states (comma-separated): q0,q1,q2
Enter alphabet (comma-separated): a,b
Enter transitions in format 'state:input => next_state' (type 'done' to finish):
q0:epsilon=>q1
q0:a=>q0
q1:b=>q1
q1:b=>q2
q2:a=>q2
done
Enter start state: q0
Enter accepting states (comma-separated): q2
NFA without epsilon transitions:
q0 --a--> {'q0'}
q0 --b--> {'q2', 'q1'}
q1 --b--> {'q2', 'q1'}
q2 --a--> {'q2'}
Final states: {'q2'}
exp - 4 :
import json
from collections import defaultdict
def read_fa():
states = input("Enter states (comma-separated): ").split(',')
states = [state.strip() for state in states]
alphabet = input("Enter alphabet (comma-separated): ").split(',')
alphabet = [symbol.strip() for symbol in alphabet]
transitions = {}
print("Enter transitions in format 'state:input => next_state' (type 'done' to finish):")
while True:
line = input().strip()
if line.lower() == "done":
break
try:
state_input, next_state = line.split('=>')
state, input_symbol = state_input.split(':')
state, input_symbol, next_state = state.strip(), input_symbol.strip(),
next_state.strip()
transitions.setdefault(state, {}).setdefault(input_symbol, set()).add(next_state)
except ValueError:
print("Invalid format. Use 'state:input => next_state'.")
start_state = input("Enter start state: ").strip()
accepting_states = set(input("Enter accepting states (comma-separated): ").split(','))
accepting_states = {state.strip() for state in accepting_states}
return {"states": states, "alphabet": alphabet, "transitions": transitions, "start_state":
start_state, "accepting_states": accepting_states}
def epsilon_closure(nfa, states):
stack, closure = list(states), set(states)
while stack:
state = stack.pop()
for next_state in nfa.get(state, {}).get('蔚', []):
if next_state not in closure:
closure.add(next_state)
stack.append(next_state)
return frozenset(closure)
def move(nfa, states, symbol):
return frozenset({s for state in states for s in nfa.get(state, {}).get(symbol, [])})
def nfa_to_dfa(nfa, start, final, alphabet):
start_closure, dfa, unprocessed = epsilon_closure(nfa, {start}), {},
[epsilon_closure(nfa, {start})]
dfa_states, dfa_finals = {start_closure}, set()
while unprocessed:
state = unprocessed.pop()
dfa[state] = {}
for symbol in alphabet - {'蔚'}:
next_state = epsilon_closure(nfa, move(nfa, state, symbol))
if next_state:
dfa[state][symbol] = next_state
if next_state not in dfa_states:
dfa_states.add(next_state)
unprocessed.append(next_state)
dfa_finals = {s for s in dfa_states if s & final}
return dfa, start_closure, dfa_finals
nfa_data = read_fa()
nfa = defaultdict(lambda: defaultdict(set), nfa_data["transitions"])
alphabet = set(nfa_data["alphabet"])
dfa, start_state, final_states = nfa_to_dfa(nfa, nfa_data["start_state"],
nfa_data["accepting_states"], alphabet)
print("DFA Transitions:")
for state, transitions in dfa.items():
for symbol, next_state in transitions.items():
print(f"{set(state)} -- {symbol} --> {set(next_state)}")
print("Start State:", set(start_state))
print("Final States:", [set(state) for state in final_states])
input :
Enter states (comma-separated): q0,q1,q2
Enter alphabet (comma-separated): a,b
Enter transitions in format 'state:input => next_state' (type 'done' to finish):
q0:a=>q0
q0:b=>q0
q0:a=>q1
q1:b=>q2
done
Enter start state: q0
Enter accepting states (comma-separated): q2
DFA Transitions:
{'q0'} -- a --> {'q1', 'q0'}
{'q0'} -- b --> {'q0'}
{'q1', 'q0'} -- a --> {'q1', 'q0'}
{'q1', 'q0'} -- b --> {'q2', 'q0'}
{'q2', 'q0'} -- a --> {'q1', 'q0'}
{'q2', 'q0'} -- b --> {'q0'}
Start State: {'q0'}
Final States: [{'q2', 'q0'}]
exp - 5 :
import re
def classify_token(token):
"""Classifies a token based on its type."""
if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
if token in {"int", "float", "char", "return", "if", "else", "while", "for", "void", "main"}:
return "KEYWORD"
return "IDENTIFIER"
elif re.match(r'^\d+$', token):
return "NUMBER"
elif token in {"+", "-", "*", "/", "%", "=", "==", "!=", "<", "<=", ">", ">=", "&&", "||", "!"}:
return "OPERATOR"
elif token in {";", ",", "{", "}", "(", ")"}:
return "DELIMITER"
elif token.startswith("\"") and token.endswith("\""):
return "STRING_LITERAL"
elif token.startswith("'") and token.endswith("'"):
return "CHAR_LITERAL"
return "UNKNOWN"
def tokenize_c_program(file_path):
with open(file_path, 'r') as file:
c_program = file.read()
token_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|\".*?\"|\'.*?\'|[+\-*/%!=<>&|]+|[{}();,]'
tokens = re.findall(token_pattern, c_program)
token_stream = [(token, classify_token(token)) for token in tokens]
return token_stream
if __name__ == "__main__":
file_path = "program.c" # Replace with the path to your C program file
token_stream = tokenize_c_program(file_path)
for token, token_name in token_stream:
print(f"{token_name}")
exp - 6 :
# Python program to compute the FIRST set for each state in the given grammar
dynamically
def compute_first(symbol, grammar, first_sets):
epsilon_symbol = '#'
'
if symbol in first_sets:
return first_sets[symbol]
first = set()
# If the symbol is a terminal
if symbol not in grammar:
if symbol == epsilon_symbol:
first.add(epsilon_symbol)
else:
first.add(symbol)
first_sets[symbol] = first
return first
for production in grammar[symbol]:
symbols = production.split()
all_have_epsilon = True # Track if all symbols in production derive epsilon
for s in symbols:
temp_first = compute_first(s, grammar, first_sets)
first.update(temp_first - {epsilon_symbol})
if epsilon_symbol not in temp_first:
all_have_epsilon = False
break # Stop if current symbol doesn't derive epsilon
if all_have_epsilon:
first.add(epsilon_symbol)
first_sets[symbol] = first
return first
def main():
grammar = {}
first_sets = {}ction rules: "))
for _ in range(n):
production = input("Enter production (format: A -> B C | D): ").strip()
head, bodies = production.split("->")
head = head.strip()
productions = [body.strip() for body in bodies.split("|")]
# Taking grammar input dynamically
n = int(input("Enter the number of produ
grammar[head] = productions
print("\nFIRST sets for all states:")
for non_terminal in grammar:
first = compute_first(non_terminal, grammar, first_sets)
print(f"FIRST({non_terminal}) = {sorted(first)}")
if __name__ == "__main__":
main()
exp - 7 :
from collections import defaultdict
grammar = defaultdict(list)
first = defaultdict(set)
follow = defaultdict(set)
def compute_first(symbol):
if symbol in first and first[symbol]:
return first[symbol]
if not symbol.isupper(): # Terminal (like 'id', '+', '*', etc.)
first[symbol].add(symbol)
return first[symbol]
for production in grammar[symbol]:
if not production or production == '#':
first[symbol].add('#')
else:
for char in production:
char_first = compute_first(char)
first[symbol].update(char_first - {'#'})
if '#' not in char_first:
break
else:
first[symbol].add('#')
return first[symbol]
def compute_follow(symbol, start_symbol):
if symbol == start_symbol:
follow[symbol].add('$')
for lhs, productions in grammar.items():
for production in productions:
for index, char in enumerate(production):
if char == symbol:
if index + 1 < len(production):
next_char = production[index + 1]
next_first = compute_first(next_char)
follow[symbol].update(next_first - {'#'})
if '#' in next_first:
if lhs != symbol:
follow[symbol].update(compute_follow(lhs,
start_symbol))
else:
if lhs != symbol:
follow[symbol].update(compute_follow(lhs, start_symbol))
return follow[symbol]
def parse_production(rhs):
tokens = []
token = ""
for char in rhs:
if char.isalnum(): # For multi-character tokens like 'id'
token += char
else:
if token:
tokens.append(token)
token = ""
if char.strip(): # Ignore whitespace
tokens.append(char)
if token:
tokens.append(token)
return tokens
def main():
n = int(input("Enter number of productions: "))
print("Enter productions (E.g., E->T R or R-># for epsilon):")
start_symbol = None
for _ in range(n):
production = input().strip()
lhs, rhs = production.split("->")
lhs = lhs.strip()
if start_symbol is None:
start_symbol = lhs
for prod in rhs.split("|"):
grammar[lhs].append(parse_production(prod.strip()))
for non_terminal in grammar:
compute_first(non_terminal)
for non_terminal in grammar:
compute_follow(non_terminal, start_symbol)
print("\nFIRST sets:")
for non_terminal in grammar:
print(f"FIRST({non_terminal}): {first[non_terminal]}")
print("\nFOLLOW sets:")
for non_terminal in grammar:
print(f"FOLLOW({non_terminal}): {follow[non_terminal]}")
if __name__ == "__main__":
main()
exp - 8:
def read_fa():
states = [s.strip() for s in input("Enter states (comma-separated): ").split(',')]
alphabet = [a.strip() for a in input("Enter alphabet (comma-separated): ").split(',')]
transitions = {s: {} for s in states}
print("Enter transitions in format 'state:input => next_state' (type 'done' to finish):")
while True:
line = input().strip()
if line.lower() == 'done':
break
state_input, next_state = map(str.strip, line.split('=>'))
state, input_symbol = map(str.strip, state_input.split(':'))
transitions[state][input_symbol] = next_state
start_state = input("Enter start state: ").strip()
accepting_states = {s.strip() for s in input("Enter accepting states (comma-separated):
").split(',')}
return states, alphabet, transitions, start_state, accepting_states
def minimize_dfa(states, alphabet, transitions, start_state, accepting_states):
partitions = [accepting_states, set(states) - accepting_states]
while True:
new_partitions = []
for group in partitions:
sub_groups = {}
for state in group:
signature = tuple(next(i for i, p in enumerate(partitions) if
transitions[state].get(a) in p) for a in alphabet)
sub_groups.setdefault(signature, set()).add(state)
new_partitions.extend(sub_groups.values())
if new_partitions == partitions:
break
partitions = new_partitions
state_mapping = {s: ''.join(sorted(group)) for group in partitions for s in group}
minimized_transitions = {state_mapping[s]: {a: state_mapping[transitions[s][a]] for a in
alphabet} for s in states}
return set(state_mapping.values()), alphabet, minimized_transitions,
state_mapping[start_state], {state_mapping[s] for s in accepting_states}
states, alphabet, transitions, start_state, accepting_states = read_fa()
minimized_fa = minimize_dfa(states, alphabet, transitions, start_state, accepting_states)
print("Minimized DFA:", {
"states": minimized_fa[0],
"alphabet": minimized_fa[1],
"transitions": minimized_fa[2],
"start_state": minimized_fa[3],
"accepting_states": minimized_fa[4]
})
input and output:
Enter states (comma-separated): a,b,c,d,e,f
Enter alphabet (comma-separated): 0,1
Enter transitions in format 'state:input => next_state' (type 'done' to finish):
a:0=>b
a:1=>c
b:0=>a
b:1=>d
c:0=>e
c:1=>f
d:0=>e
d:1=>f
e:0=>e
e:1=>f
f:0=>f
f:1=>f
done
Enter start state: a
Enter accepting states (comma-separated): c,d,e
Minimized DFA: {'states': {'cde', 'ab', 'f'}, 'alphabet': ['0', '1'], 'transitions': {'ab': {'0': 'ab', '1':
'cde'}, 'cde': {'0': 'cde', '1': 'f'}, 'f': {'0': 'f', '1': 'f'}}, 'start_state': 'ab', 'accepting_states': {'cde'}}
