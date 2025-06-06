Implementing error reporting and recovery mechanisms in a C++ parser is crucial for improving user experience, as it allows the parser to handle syntax errors gracefully without crashing. Below is an example demonstrating how to enhance the previous Flex and Bison parser with error reporting and recovery mechanisms.

### Enhancing the Bison Parser with Error Handling

Here’s an updated version of the `parser.y` file, which includes error handling for syntax errors:

```c
// parser.y
%{
#include <stdio.h>
#include <stdlib.h>

// Error handling function declaration
void yyerror(const char *s);
int yylex();
%}

%union {
    int val;
}

%token <val> NUMBER
%token PLUS MINUS MULT DIV LPAREN RPAREN

%type <val> expr

%%

// Grammar rules
expr:
      expr PLUS expr { $$ = $1 + $3; }
    | expr MINUS expr { $$ = $1 - $3; }
    | expr MULT expr { $$ = $1 * $3; }
    | expr DIV expr { 
        if ($3 == 0) {
            yyerror("Error: Division by zero");
            $$ = 0; // Return a default value or handle as needed
        } else {
            $$ = $1 / $3; 
        }
    }
    | LPAREN expr RPAREN { $$ = $2; }
    | NUMBER { $$ = $1; }
    ;

%%

// Error handling function
void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}

// Main function
int main() {
    printf("Enter an expression: ");
    while (yyparse() != 0) {
        printf("Try again: ");
    }
    return 0;
}
```

### Explanation of the Changes

1. **Error Reporting**:
   - The `yyerror` function is used to report errors. When a syntax error occurs, this function is called, and a meaningful error message is printed to `stderr`.

2. **Error Recovery**:
   - In the `expr` rules, we can provide specific error handling for certain conditions. For example, in the division rule, we check for division by zero. If it occurs, we report the error and return a default value.

3. **Looping on Errors**:
   - In the `main` function, after calling `yyparse()`, if it returns a non-zero value indicating an error, the program prompts the user to enter the expression again, allowing for recovery without crashing.

### Implementing Recovery Strategies

To further enhance error recovery, we can introduce mechanisms to skip erroneous tokens and continue parsing:

```c
%%

// Grammar rules
expr:
      expr PLUS expr { $$ = $1 + $3; }
    | expr MINUS expr { $$ = $1 - $3; }
    | expr MULT expr { $$ = $1 * $3; }
    | expr DIV expr {
        if ($3 == 0) {
            yyerror("Error: Division by zero");
            $$ = 0; // Handle as needed
        } else {
            $$ = $1 / $3; 
        }
    }
    | LPAREN expr RPAREN { $$ = $2; }
    | NUMBER { $$ = $1; }
    ;

%%

// Error recovery rule
error:
    { 
        fprintf(stderr, "Syntax error encountered. Attempting to recover...\n");
        yyerror("Syntax error");
        yyclearin;  // Clear the current token
        while (yylex() != ';') { // Skip until a semicolon or a newline
            // Discard erroneous input
        }
        // Continue parsing
    }

int main() {
    printf("Enter an expression: ");
    while (yyparse() != 0) {
        printf("Try again: ");
    }
    return 0;
}
```

### Key Points of the Recovery Mechanism

1. **Error Rule**: 
   - An `error` rule can be defined to handle syntax errors. When an error is encountered, this rule can trigger, allowing the parser to perform specific recovery actions.

2. **Skipping Tokens**: 
   - In the recovery block, the `yylex()` function is called in a loop to discard tokens until a semicolon (`;`) or newline is encountered. This allows the parser to ignore the erroneous input and continue processing.

3. **Informative Feedback**: 
   - The user receives feedback about syntax errors, enhancing the user experience by informing them what went wrong.

### Conclusion

With these enhancements, the parser can gracefully handle syntax errors, providing informative error messages and allowing users to continue entering input without crashing the program. This is a critical aspect of building robust language processing tools, making it easier for users to interact with the system and identify issues in their input.