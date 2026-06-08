```markdown
# sensorforge Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the `sensorforge` Rust codebase. You'll learn about file naming, import/export styles, commit message habits, and how to structure and run tests. This guide is ideal for contributors looking to quickly align with the project's standards.

## Coding Conventions

### File Naming
- **Style:** camelCase
- **Example:**  
  - `sensorManager.rs`
  - `dataParser.rs`

### Import Style
- **Style:** Relative imports are preferred.
- **Example:**
  ```rust
  mod sensorManager;
  use crate::dataParser::parseData;
  ```

### Export Style
- **Style:** Named exports.
- **Example:**
  ```rust
  pub fn initialize_sensor() { /* ... */ }
  pub struct SensorConfig { /* ... */ }
  ```

### Commit Messages
- **Pattern:** Freeform, average length ~70 characters.
- **Prefixes:** None enforced.
- **Example:**  
  `Add support for new temperature sensor model`

## Workflows

### Adding a New Module
**Trigger:** When you need to add a new feature or component.
**Command:** `/add-module`

1. Create a new file using camelCase (e.g., `newFeature.rs`).
2. Implement your module logic.
3. Use relative imports to include other modules as needed.
4. Export public functions or structs with `pub`.
5. Add tests in a corresponding `*.test.*` file.

### Updating an Existing Module
**Trigger:** When modifying or extending functionality.
**Command:** `/update-module`

1. Locate the module file (e.g., `sensorManager.rs`).
2. Make your changes following the code style conventions.
3. Update or add tests if necessary.
4. Commit changes with a clear, descriptive message.

### Writing and Running Tests
**Trigger:** When validating new or changed code.
**Command:** `/run-tests`

1. Create or update test files matching the pattern `*.test.*`.
2. Write tests using the project's preferred (unknown) framework.
3. Run tests using the appropriate Rust test command (e.g., `cargo test`).
4. Ensure all tests pass before committing.

## Testing Patterns

- **File Pattern:** Test files are named with the pattern `*.test.*`, such as `sensorManager.test.rs`.
- **Framework:** Not explicitly specified; use standard Rust testing where unsure.
- **Example:**
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_initialize_sensor() {
          assert!(initialize_sensor().is_ok());
      }
  }
  ```

## Commands
| Command        | Purpose                                   |
|----------------|-------------------------------------------|
| /add-module    | Scaffold a new module with conventions    |
| /update-module | Update an existing module                 |
| /run-tests     | Run all tests in the codebase             |
```
