const fs = require('fs');
const content = fs.readFileSync('frontend/src/pages/WorkspacePage.tsx', 'utf8');
let openBraces = 0;
let openParens = 0;
let inString = false;
let stringChar = '';
for (let i = 0; i < content.length; i++) {
  const char = content[i];
  if (!inString && (char === '"' || char === "'" || char === '`')) {
    inString = true;
    stringChar = char;
  } else if (inString && char === stringChar && content[i-1] !== '\\') {
    inString = false;
    stringChar = '';
  }
  if (!inString) {
    if (char === '{') openBraces++;
    if (char === '}') openBraces--;
    if (char === '(') openParens++;
    if (char === ')') openParens--;
  }
  if (openBraces < 0 || openParens < 0) {
    console.log('Negative at position', i, 'char:', char, 'braces:', openBraces, 'parens:', openParens);
  }
}
console.log('Final - braces:', openBraces, 'parens:', openParens);