const fs = require('fs');
const content = fs.readFileSync('frontend/src/pages/WorkspacePage.tsx', 'utf8');
const lines = content.split('\n');

function inString(line, pos) {
  let inString = false;
  let stringChar = '';
  for (let i = 0; i < pos; i++) {
    const char = line[i];
    if (!inString && (char === '"' || char === "'" || char === '`')) {
      inString = true;
      stringChar = char;
    } else if (inString && char === stringChar && line[i-1] !== '\\') {
      inString = false;
    }
  }
  return inString;
}

let parenDepth = 0;
let openBrace = 0;
let openBracket = 0;
let unmatchedParenLines = [];
let unmatchedBraceLines = [];
for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  for (let j = 0; j < line.length; j++) {
    const char = line[j];
    if (char === '(' && !inString(line, j)) {
      if (parenDepth === 0) unmatchedParenLines.push({line: i+1, col: j+1, context: line.substring(Math.max(0, j-20), j+20)});
      parenDepth++;
    }
    if (char === ')' && !inString(line, j)) {
      if (parenDepth > 0) parenDepth--;
      else console.log('Extra ) at line', i+1, 'col', j+1);
    }
    if (char === '{' && !inString(line, j)) {
      if (openBrace === 0) unmatchedBraceLines.push({line: i+1, col: j+1, context: line.substring(Math.max(0, j-20), j+20)});
      openBrace++;
    }
    if (char === '}' && !inString(line, j)) {
      if (openBrace > 0) openBrace--;
      else console.log('Extra } at line', i+1, 'col', j+1);
    }
    if (char === '[' && !inString(line, j)) openBracket++;
    if (char === ']' && !inString(line, j)) openBracket--;
  }
}
console.log('Final - paren:', parenDepth, 'brace:', openBrace, 'bracket:', openBracket);
console.log('Unmatched paren locations:', unmatchedParenLines);
console.log('Unmatched brace locations:', unmatchedBraceLines);