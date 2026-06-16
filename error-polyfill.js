// Patch: wrap err.message assignment in try-catch to survive getter-only message
const fs = require('fs');
const path = '/app/node_modules/.pnpm/next@14.2.21_@babel+core@7.24.3_@opentelemetry+api@1.9.0_@playwright+test@1.47.2_babel-plugin_yqlzwwxnhgfuesbfmfmtmhekse/node_modules/next/dist/server/next-server.js';

let content = fs.readFileSync(path, 'utf8');
const old = "err.message = `An error occurred while loading instrumentation hook: ${err.message}`;";
const rep = "try { err.message = `An error occurred while loading instrumentation hook: ${err.message}`; } catch(e) {}";

if (content.includes(old)) {
  content = content.replace(old, rep);
  fs.writeFileSync(path, content);
}
