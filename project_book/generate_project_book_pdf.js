const fs = require('fs'), path=require('path');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:900,height:1200}});
 await page.goto('file:///'+path.join(__dirname,'project_book.html').replace(/\\/g,'/'),{waitUntil:'networkidle'});
 await page.evaluate(()=>document.fonts.ready);
 const audit=await page.evaluate(()=>Array.from(document.querySelectorAll('.page')).map((p,i)=>({page:i+1,contentBottom:Math.max(...Array.from(p.querySelectorAll('.bodycontent,.covermeta')).map(x=>x.getBoundingClientRect().bottom-p.getBoundingClientRect().top)),footerTop:p.querySelector('.foot').getBoundingClientRect().top-p.getBoundingClientRect().top,overflow:p.scrollHeight>p.clientHeight})));
 fs.writeFileSync(path.join(__dirname,'layout_audit.json'),JSON.stringify(audit,null,2));
 console.log(audit);
 if(audit.some(a=>a.contentBottom>a.footerTop-12 || a.overflow)) throw new Error('Page layout overflow; repair before PDF export');
 await page.pdf({path:path.join(__dirname,'245-赛题5.pdf'),preferCSSPageSize:true,printBackground:true});
 await browser.close();
})();
