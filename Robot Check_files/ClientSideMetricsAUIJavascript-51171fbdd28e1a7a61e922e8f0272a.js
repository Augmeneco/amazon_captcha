(function(a,c,p){function l(a){for(var b={},f,c,d=0;d<a.length;d++)c=a[d],f=c.r+c.s+c.m,c.c&&(b[f]||(b[f]=[]),b[f].push(a[d]));return b}function k(a){for(var b=1;b<arguments.length;b++){var f=arguments[b];try{if(f.isSupported)return f.send(a)}catch(c){}}}function j(){for(var a=0;a<w.length;a++)w[a]();u.length&&k(l(u.splice(0,u.length)),F,G,A);D=z=0}function e(b,f,e){e=e||{};0===e.bf&&d.isBF||(b={r:e.r||d.rid,s:e.s||a.ue_sid,m:e.m||a.ue_mid,mkt:e.mkt||a.ue_mkt,sn:e.sn||a.ue_sn,c:f,d:b,t:e.t||d.d(),
cs:e.c&&a.ue_qsl},e.b?k(l([b]),F,A):e.nb?k(l([b]),F,G,A):e.img||I[f]?k(l([b]),A):e.n?(u.push(b),0===B?j():D||(D=c.setTimeout(j,B))):(u.push(b),z||(z=c.setTimeout(j,H))))}function n(a,b,f){E++;E==s?e({m:"Max number of Forester Logs exceeded",f:"forester-client.js",logLevel:"ERROR"},c.ue_err_chan||"jserr"):E<s&&e(a,b,f)}function i(){if(!y){for(var a=0;a<x.length;a++)x[a]();for(a=0;a<w.length;a++)w[a]();k(l(u.splice(0,u.length)),F,A);y=!0}}var g={};(function(){function a(b){return 10>b?"0"+b:b}function b(a){c.lastIndex=
0;return c.test(a)?'"'+a.replace(c,function(a){var b=h[a];return"string"===typeof b?b:"\\u"+("0000"+a.charCodeAt(0).toString(16)).slice(-4)})+'"':'"'+a+'"'}function f(a,c){var g,h,k,j,l=d,C,q=c[a];q&&"object"===typeof q&&"function"===typeof q.toJSON&&(q=q.toJSON(a));"function"===typeof i&&(q=i.call(c,a,q));switch(typeof q){case "string":return b(q);case "number":return isFinite(q)?String(q):"null";case "boolean":case "null":return String(q);case "object":if(!q)return"null";d+=e;C=[];if("[object Array]"===
Object.prototype.toString.apply(q)){j=q.length;for(g=0;g<j;g+=1)C[g]=f(g,q)||"null";k=0===C.length?"[]":d?"[\n"+d+C.join(",\n"+d)+"\n"+l+"]":"["+C.join(",")+"]";d=l;return k}if(i&&"object"===typeof i)for(j=i.length,g=0;g<j;g+=1)"string"===typeof i[g]&&(h=i[g],(k=f(h,q))&&C.push(b(h)+(d?": ":":")+k));else for(h in q)Object.prototype.hasOwnProperty.call(q,h)&&(k=f(h,q))&&C.push(b(h)+(d?": ":":")+k);k=0===C.length?"{}":d?"{\n"+d+C.join(",\n"+d)+"\n"+l+"}":"{"+C.join(",")+"}";d=l;return k}}"function"!==
typeof Date.prototype.toJSON&&(Date.prototype.toJSON=function(){return isFinite(this.valueOf())?this.getUTCFullYear()+"-"+a(this.getUTCMonth()+1)+"-"+a(this.getUTCDate())+"T"+a(this.getUTCHours())+":"+a(this.getUTCMinutes())+":"+a(this.getUTCSeconds())+"Z":null},String.prototype.toJSON=Number.prototype.toJSON=Boolean.prototype.toJSON=function(){return this.valueOf()});var c=/[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,d,
e,h={"":"\\b","\t":"\\t","\n":"\\n","":"\\f","\r":"\\r",'"':'\\"',"\\":"\\\\"},i;"function"!==typeof g.stringify&&(g.stringify=function(a,b,c){var g;e=d="";if("number"===typeof c)for(g=0;g<c;g+=1)e+=" ";else"string"===typeof c&&(e=c);if((i=b)&&"function"!==typeof b&&("object"!==typeof b||"number"!==typeof b.length))throw Error("JSON.stringify");return f("",{"":a})})})();var h=function(){function a(b,f){if(null==b)return f.push("!n");if("number"===typeof b)return f.push("!"+b);if("string"===typeof b)return"\\"==
b[b.length-1]?f.push("'"+b.replace(/'/g,"\\'")+"u005C'"):f.push("'"+b.replace(/'/g,"\\'")+"'");if("boolean"===typeof b)return f.push(b?"!t":"!f");if(b instanceof Array){f.push("*");for(var c=0;c<b.length;c++)a(b[c],f);return f.push(")")}if("object"==typeof b){f.push("(");for(c in b)b.hasOwnProperty(c)&&(f.push(c),a(b[c],f));return f.push(")")}return f.push("!n")}return{stringify:function(b){var f=[];a(b,f);return f.join("")}}}(),t=a.ue_qsl||2E3,s=1E3,o=function(){},b="",q=c.JSON&&c.JSON.stringify||
g&&g.stringify,f=h.stringify,d=a.ue||{},h=a.uet||o;(a.uet||o)("bb","ue_frst_v2",{wb:1});var r="//"+a.ue_furl+"/1/batch/1/OE/",u=[],x=[],w=[],y=!1,D,z,B=void 0===a.ue_hpfi?1E3:a.ue_hpfi,H=void 0===a.ue_lpfi?1E4:a.ue_lpfi,I={"scheduled-delivery":1},E=0,G=function(){function f(){if(c.XDomainRequest){var a=new XDomainRequest;a.onerror=o;a.ontimeout=o;a.onprogress=o;a.onload=o;a.timeout=0;return a}if(c.XMLHttpRequest){a=new XMLHttpRequest;if(!("withCredentials"in a))throw b;return a}if(c.ActiveXObject){for(var d=
0;d<m.length&&!a;d++)try{a=new ActiveXObject(m[d]),m=[m[d]]}catch(e){}return a}}function e(b){for(var f=[],c=b[0]||{},m=0;m<b.length;m++){var g={};g[b[m].c]=b[m].d;f.push(g)}return{rid:c.r||d.rid,sid:c.s||a.ue_sid,mid:c.m||a.ue_mid,mkt:c.mkt||a.ue_mkt,sn:c.sn||a.ue_sn,reqs:f}}var m="MSXML2.XMLHTTP.6.0 MSXML2.XMLHTTP.5.0 MSXML2.XMLHTTP.4.0 MSXML2.XMLHTTP.3.0 MSXML2.XMLHTTP Microsoft.XMLHTTP".split(" ");return{send:function(a){for(var c in a)if(a.hasOwnProperty(c)&&a[c].length){var d=e(a[c]),m=f();
if(!m)throw b;m.open("POST",r,!0);m.setRequestHeader&&m.setRequestHeader("Content-type","text/plain");m.send(q(d))}},buildPOSTBodyLog:e,isSupported:!0}}(),A=function(){return{send:function(b){for(var c in b)if(b.hasOwnProperty(c)){for(var e=b[c],g=e,h={},i=void 0,k=0;k<g.length;k++)i=g[k].c,h[i]||(h[i]=[]),h[i].push(g[k]);var e=e[0]||{},g=e.sn||a.ue_sn,e=r+(e.m||a.ue_mid)+":"+(e.s||a.ue_sid)+":"+(e.r||d.rid)+(g?":"+g:""),g=[],i=e,k=[],j=void 0;for(j in h)if(h.hasOwnProperty(j))for(var l=0;l<h[j].length;l++){var n=
h[j][l],o=encodeURIComponent((n.cs?f:q)(n.d));k.push({l:o,t:n.t,p:1,c:j,d:n.cs?"c":"j"})}h=k;k=void 0;j="$";for(n=0;n<h.length;){l=h[n];k!=l.c?(i+=j+l.c+"=",j="&",k=l.c):i+=",";var o=i,s=l.d+":",p=l,i=(p.l.match(".{1,"+(t-i.length)+"}[^%]{0,2}")||[])[0]||"";p.l=p.l.substr(i.length);i=o+(s+i+":"+l.t);if(l.l)i+=":"+l.p++ +"_",g.push(i),i=e,j="$",k=0;else if(n++,1!=l.p)for(i+=":"+l.p+"_"+l.p,o=0;o<l.p-1;o++)g[g.length-o-1]+=l.p}g.push(i);e=g;for(g=0;g<e.length;g++)(new Image).src=e[g]}},isSupported:!0}}(),
F=function(){return{send:function(a){for(var f in a)if(a.hasOwnProperty(f)){var c=G.buildPOSTBodyLog(a[f]);if(!navigator.sendBeacon(r,q(c)))throw b;}},isSupported:!!navigator.sendBeacon}}();d._fic=A;d._fac=G;d._fbc=F;d._flq=u;d.sid=d.sid||a.ue_sid;d.mid=d.mid||a.ue_mid;d.furl=d.furl||a.ue_furl;d.sn=d.sn||a.ue_sn;d.isBF=function(){var a=c.performance||c.webkitPerformance,b=p.ue_backdetect&&p.ue_backdetect.ue_back&&document.ue_backdetect.ue_back.value,f=d.bfini;return a&&a.navigation&&2===a.navigation.type||
1<f||!f&&1<b}();try{c.amznJQ&&c.amznJQ.declareAvailable&&c.amznJQ.declareAvailable("forester-client"),c.P&&c.P.register&&c.P.register("forester-client",o)}catch(O){a.ueLogError(O,{logLevel:"WARN"})}(function(){d.log&&d.log.isStub&&(d.log.replay(function(a,b,f){var c=a[2]||{};c.t=b;c.r=f;c.n=1;n(a[0],a[1],c)}),d.onunload.replay(function(a){x.push(a[0])}),d.onflush.replay(function(a){w.push(a[0])}))})();d.log=n;d.log.reset=function(){E=0};d.onunload=function(a){x.push(a)};d.onflush=function(a){w.push(a)};
d.attach("beforeunload",i);d.attach("pagehide",i);h("ld","ue_frst_v2",{wb:1})})(ue_csm,window,document);
(function(a,c){function p(a){if(a)return a.replace(/^\s+|\s+$/g,"")}function l(b,c){if(!b)return{};b.m&&b.m.message&&(b=b.m);var f=c.m||c.message||"",f=b.m&&b.m.message?f+b.m.message:b.m&&b.m.target&&b.m.target.tagName?f+("Error handler invoked by "+b.m.target.tagName+" tag"):b.m?f+b.m:b.message?f+b.message:f+"Unknown error",f={m:f,f:b.f||b.sourceURL||b.fileName||b.filename||b.m&&b.m.target&&b.m.target.src,l:b.l||b.line||b.lineno||b.lineNumber,c:b.c?""+b.c:b.c,s:[],t:a.ue.d(),name:b.name,type:b.type,
csm:g+" "+(b.fromOnError?"onerror":"ueLogError")},d,e,j=0,l=0,n;d=b.stack||(b.err?b.err.stack:"");f.pageURL=c.pageURL||""+(window.location?window.location.href:"")||"missing";f.logLevel=c.logLevel||i;if(e=c.attribution)f.attribution=""+e;if(d&&d.split)for(f.csm+=" stack",e=d.split("\n");j<e.length&&f.s.length<h;)(d=e[j++])&&f.s.push(p(d));else for(f.csm+=" callee",e=k(b.args||arguments,"callee"),l=j=0;e&&j<h;)n=t,e.skipTrace||(d=e.toString())&&d.substr&&(n=0===l?4*t:n,n=1==l?2*t:n,f.s.push(d.substr(0,
n)),l++),e=k(e,"caller"),j++;if(!f.f&&0<f.s.length&&(j=f)&&j.s){var y;d=0<j.s.length?j.s[0]:"";e=1<j.s.length?j.s[1]:"";d&&(y=d.match(o));y&&3==y.length||!e||(y=e.match(s));y&&3==y.length&&(j.f=y[1],j.l=y[2])}return f}function k(a,c){try{return a[c]}catch(f){}}function j(b,e){if(b){var f=l(b,e);a.ue.log(f,e.channel||n,{nb:1});try{var d=c.console,g=c.JSON,h="Error logged: ";if(d){if(g&&g.stringify)try{h+=g.stringify(f)}catch(i){h+="no info provided; converting to string failed"}else h+=f.m;"function"===
typeof d.error?d.error(h,f):"function"===typeof d.log&&d.log(h,f)}}catch(k){}}}function e(b,c){if(b&&!(a.ue_err.ec>a.ue_err.mxe)){a.ue_err.ec++;a.ue_err.ter.push(b);var c=c||{},f=b.logLevel||c.logLevel;c.logLevel=f;c.attribution=b.attribution||c.attribution;f&&f!=i||ue_err.ecf++;j(b,c)}}if(!a.ueLogError||a.ueLogError.isStub){var n=a.ue_err_chan||"jserr",i="FATAL",g="v5",h=20,t=256,s=/\(?([^\s]*):(\d+):\d+\)?/,o=/.*@(.*):(\d*)/;j.skipTrace=1;l.skipTrace=1;e.skipTrace=1;(function(){if(a.ue_err.erl){var b=
a.ue_err.erl.length,c,f;for(c=0;c<b;c++)f=a.ue_err.erl[c],j(f.ex,f.info);ue_err.erl=[]}})();a.ueLogError=e}})(ue_csm,window);
(function(a,c){a.ue_cel||(a.ue_cel=function(){function p(a,c){c?c.r=s:c={r:s,c:1};c.clog&&g.clog?g.clog(a,n,c):c.glog&&g.glog?g.glog(a,n,c):g.log(a,n,c)}function l(){var a=j.length;if(0<a){for(var c=[],f=0;f<a;f++){var d=j[f].api;d.ready()?(d.on({ts:g.d,ns:n}),e.push(j[f]),p({k:"mso",n:j[f].name,t:g.d()})):c.push(j[f])}j=c}}function k(){if(!k.executed){for(var a=0;a<e.length;a++)e[a].api.off&&e[a].api.off({ts:g.d,ns:n});p({k:"eod",t0:g.t0,t:g.d()},{c:1});k.executed=1;for(a=0;a<e.length;a++)j.push(e[a]);
e=[];clearTimeout(i)}}var j=[],e=[],n=a.ue_cel_ns||"cel",i,g=a.ue,h=a.uet,t=a.uex,s=g.rid,o=c.requestAnimationFrame||function(a){a()};if(g.isBF)p({k:"bft",t:g.d()});else return"function"==typeof h&&h("bb","csmCELLSframework",{wb:1}),setTimeout(l,0),g.onunload(k),i=setTimeout(k,6E5),"function"==typeof t&&t("ld","csmCELLSframework",{wb:1}),{registerModule:function(a,c){j.push({name:a,api:c});p({k:"mrg",n:a,t:g.d()});l()},reset:function(a){p({k:"rst",t0:g.t0,t:g.d()});j=j.concat(e);e=[];for(var c=j.length,
f=0;f<c;f++)j[f].api.off(),j[f].api.reset();s=a||g.rid;l();clearTimeout(i);i=setTimeout(k,6E5);k.executed=0},timeout:function(a,e){return c.setTimeout(function(){o(a)},e)},log:p}}())})(ue_csm,window);
(function(a,c,p){a.ue_pdm||ue.isBF||(a.ue_pdm=function(){function l(){var b={w:e.width,aw:e.availWidth,h:e.height,ah:e.availHeight,cd:e.colorDepth,pd:e.pixelDepth},c={w:p.body.scrollWidth,h:p.body.scrollHeight};h&&h.w==b.w&&h.h==b.h&&h.aw==b.aw&&h.ah==b.ah&&h.pd==b.pd&&h.cd==b.cd||(h=b,h.t=i(),h.k="sci",o(h));t&&t.w==c.w&&t.h==c.h||(t=c,t.t=i(),t.k="doi",o(t));n=a.ue_cel.timeout(l,g)}function k(){o({k:"ebl",t:i()})}function j(){o({k:"efo",t:i()})}var e,n,i,g,h,t,s=a.ue,o=a.ue_cel.log,b=a.uet,q=a.uex;
"function"==typeof b&&b("bb","csmCELLSpdm",{wb:1});return{on:function(b){g=b.timespan||500;i=b.ts;e=c.screen;s.attach&&(s.attach("blur",k,c),s.attach("focus",j,c));b=c.location;o({k:"pmd",o:b.origin,p:b.pathname,t:i()});a.ue_cel.timeout(l,0);"function"==typeof q&&q("ld","csmCELLSpdm",{wb:1})},off:function(){clearTimeout(n);s.detach&&(s.detach("blur",k,c),s.detach("focus",j,c));s.count&&(s.count("cel.PDM.TotalExecutions",0),s.count("cel.PDM.TotalExecutionTime",0),s.count("cel.PDM.AverageExecutionTime",
0/0))},ready:function(){return p.body&&a.ue_cel&&a.ue_cel.log},reset:function(){h=t=null}}}(),a.ue_cel&&a.ue_cel.registerModule("page module",a.ue_pdm))})(ue_csm,window,document);
(function(a,c){a.ue_vpm||ue.isBF||(a.ue_vpm=function(){function p(){var a=n(),b={w:c.innerWidth,h:c.innerHeight,x:c.pageXOffset,y:c.pageYOffset};k&&k.w==b.w&&k.h==b.h&&k.x==b.x&&k.y==b.y||(b.t=a,b.k="vpi",k=b,t(k,{clog:1}));j=0;i=n()-a;g+=1}function l(){j||(j=a.ue_cel.timeout(p,e))}var k,j,e,n,i=0,g=0,h=a.ue,t=a.ue_cel.log,s=a.uet,o=a.uex,b=h.attach,q=h.detach;"function"==typeof s&&s("bb","csmCELLSvpm",{wb:1});return{on:function(c){n=c.ts;e=c.timespan||100;a.ue_cel.timeout(p,0);b&&(b("scroll",l),
b("resize",l));"function"==typeof o&&o("ld","csmCELLSvpm",{wb:1})},off:function(){clearTimeout(j);q&&(q("scroll",l),q("resize",l));h.count&&(h.count("cel.VPI.TotalExecutions",g),h.count("cel.VPI.TotalExecutionTime",i),h.count("cel.VPI.AverageExecutionTime",i/g))},ready:function(){return a.ue_cel&&a.ue_cel.log},reset:function(){k=void 0},getVpi:function(){return k}}}(),a.ue_cel&&a.ue_cel.registerModule("viewport module",a.ue_vpm))})(ue_csm,window);
(function(a,c,p){var l=a.ue||{};!l.isBF&&!a.ue_fem&&p.querySelector&&c.getComputedStyle&&[].forEach&&(a.ue_fem=function(){function k(){a.ue_cel.timeout(function(){f.splice(0).forEach(function(a){d(a,{clog:1})})},0)}function j(a){for(var d={x:c.pageXOffset,y:c.pageYOffset},e=0;e<b.length;e++){var g=b[e];if(g.w&&g.w.length)for(var h=0;h<g.w.length;h++){var i=g.w[h],k;a:{try{var j=i,l=d;if(j){var n=j.getBoundingClientRect();k={x:n.left+l.x|0,y:n.top+l.y|0,w:n.width|0,h:n.height|0,d:(0===j.offsetWidth&&
0===j.offsetHeight)|0}}else k=void 0;break a}catch(q){}k=void 0}if(k&&!i.cel_b)i.cel_b=k,f.push({n:i.cel_n,w:i.cel_b.w,h:i.cel_b.h,d:i.cel_b.d,x:i.cel_b.x,y:i.cel_b.y,t:a,k:"ewi",cl:i.className});else{if(j=k)j=i.cel_b,l=k,j=l.d===j.d&&1===l.d?!1:!((j.x>l.x?3>j.x-l.x:3>l.x-j.x)&&(j.y>l.y?3>j.y-l.y:3>l.y-j.y)&&(j.w>l.w?3>j.w-l.w:3>l.w-j.w)&&(j.h>l.h?3>j.h-l.h:3>l.h-j.h)&&j.d===l.d);j&&(i.cel_b=k,f.push({n:i.cel_n,w:i.cel_b.w,h:i.cel_b.h,d:i.cel_b.d,x:i.cel_b.x,y:i.cel_b.y,t:a,k:"ewi"}))}}}}function e(a){for(var c=
0;c<b.length;c++){var d=b[c],e;e=d;e=e.c?p.getElementsByClassName(e.c):e.id?[p.getElementById(e.id)]:p.querySelectorAll(e.s);var g=d.w||[],h;for(h=0;h<g.length;h++){var j=g[h];r.contains(j)||f.push({n:j.cel_n,t:a,k:"ewd"})}d.w=[];for(h=0;h<e.length;h++)if(g=e[h])g.cel_n||(g.cel_n=g.getAttribute("cel_widget_id")||(d.id_gen||q)(g,h)||g.id),d.w.push(g)}i()}function n(){A||(A=a.ue_cel.timeout(function(){A=null;g("dwe",e);k()},h))}function i(){A||G||(G=a.ue_cel.timeout(function(){G=null;g("dwpc",j);k()},
h))}function g(a,b){var c=o();b(c);var d=o(),c=d-c;c<=t?d=0:d-F>=s?(F=d,d=0):(ue_fem.off(),d=1);(f.length||d)&&f.push({k:"ewt",e:a,d:c,ex:d,t:o()})}var h=50,t=10,s=3E3,o,b=[],q=function(){},f=[],d=a.ue_cel.log,r,u,x,w,y=c.MutationObserver||c.WebKitMutationObserver||c.MozMutationObserver,D=!!y,z,B,H="DOMAttrModified",I="DOMNodeInserted",E="DOMNodeRemoved",G,A,F=-s;"function"==typeof uet&&uet("bb","csmCELLSfem",{wb:1});return{on:function(c){function f(){if(x&&w&&r&&r.contains&&r.getBoundingClientRect&&
o){if(D){var a={attributes:!0,subtree:!0};z=new y(i);B=new y(n);z.observe(r,a);B.observe(r,{childList:!0,subtree:!0});B.observe(u,a)}else x.call(r,H,i),x.call(r,I,n),x.call(r,E,n),x.call(u,I,i),x.call(u,E,i);n()}}r=p.body;u=p.head;x=r.addEventListener;w=r.removeEventListener;o=c.ts;b=a.cel_widgets||[];l.deffered?f():l.attach&&l.attach("load",f);"function"==typeof uex&&uex("ld","csmCELLSfem",{wb:1})},off:function(){x&&w&&r&&r.contains&&r.getBoundingClientRect&&o&&(B&&(B.disconnect(),B=null),z&&(z.disconnect(),
z=null),w.call(r,H,i),w.call(r,I,n),w.call(r,E,n),w.call(u,I,i),w.call(u,E,i))},ready:function(){return a.ue_cel&&a.ue_cel.log},reset:function(){b=a.cel_widgets||[]}}}(),a.ue_cel&&a.ue_fem&&a.ue_cel.registerModule("features module",a.ue_fem))})(ue_csm,window,document);
(function(a,c){a.ue_mcm||a.ue.isBF||(a.ue_mcm=function(){function p(a){if(a.id)return"//*[@id='"+a.id+"']";var c;c=1;var e;for(e=a.previousSibling;e;e=e.previousSibling)e.nodeName==a.nodeName&&(c+=1);e=a.nodeName;1!=c&&(e+="["+c+"]");a.parentNode&&(e=p(a.parentNode)+"/"+e);return e}function l(a){var e=a.srcElement||a.target||{},l={k:k,w:c.body.scrollWidth,h:c.body.scrollHeight,t:j(),x:a.pageX,y:a.pageY,p:p(e),n:e.nodeName,e:i};a.button&&(l.b=a.button);e.href&&(l.r=e.href);e.id&&(l.i=e.id);e.className&&
(l.c=e.className.split(/\s+/));n(l,{n:1,c:1})}var k="mcm",j,e=a.ue,n=a.ue_cel.log,i=a.ue_mce||"click";return{on:function(a){j=a.ts;e.attach&&e.attach(i,l,c)},off:function(){e.detach&&e.detach(i,l,c)},ready:function(){return a.ue_cel&&a.ue_cel.log},reset:function(){}}}(),a.ue_cel&&a.ue_cel.registerModule("mouse click module",a.ue_mcm))})(ue_csm,document);
(function(a,c){a.ue_mmm||a.ue.isBF||(a.ue_mmm=function(){function p(a){h={x:a.pageX,y:a.pageY}}function l(){!h||t&&h.x==t.x&&h.y==t.y||k()}function k(){if(h){var a={k:j,t:e(),x:h.x,y:h.y};ue_cel.log(a);t=h}}var j="mmm3",e,n,i=a.ue,g,h,t;return{on:function(a){e=a.ts;n=a.ns;i.attach("mousemove",p,c);g=setInterval(l,100)},off:function(){n&&k();clearInterval(g);i.detach("mousemove",p,c)},ready:function(){return 1},reset:function(){t=h=null}}}(),a.ue_cel&&a.ue_cel.registerModule("mouse move module",a.ue_mmm))})(ue_csm,
document);
(function(a,c,p,l){!a.ue.isBF&&!p.ue_rpl&&a.MutationObserver&&a.performance&&a.performance.now&&l&&(p.ue_rpl=function(){function k(a,b,c){x.unshift({elem:a,type:b,time:c})}function j(a,b,c){w.push({elem:a,type:b,time:c})}function e(a,b){if(b)return{h:a.offsetHeight,w:a.offsetWidth}}function n(a,b){if(0===a.childNodes.length&&!b)return a.textContent}function i(a){if(a.attributes&&a.attributes.length){for(var b={},c=0;c<a.attributes.length;c++)b[a.attributes[c].name]=a.attributes[c].value;return b}}
function g(){for(var a=performance.now(),b=!1;performance.now()-a<s&&!b;){for(b=d;0<b--&&0<x.length;){var c=x.shift(),f=c.type,e=c.elem;if(e&&1==e.nodeType||3==e.nodeType||8==e.nodeType)z[f](e,c.time)}b=0===x.length}h()}function h(){!x.length&&w.length&&(x=w,w=[]);x.length&&l.timeout(g,0)}function p(a){for(var c=r(),e=0;e<a.length;e++){var d=a[e];if("attributes"==d.type)j(d.target,q,c);else{for(var g=c,i=0;i<d.addedNodes.length;i++){var k=d.addedNodes[i],l=k.parentElement||{};j(k,l.csm_sensitive_info||
l.csm_blacklist_parent?b:o,g)}for(i=0;i<d.removedNodes.length;i++)j(d.removedNodes[i],f,g)}}h()}var s=4.5,o=0,b=1,q=2,f=3,d=null,r=null,u=null,x=[],w=[],y=0,D=0,z=[function(a,c){var f=a.childNodes.length,e=a.className&&-1!=a.className.indexOf("copilot-secure-display");a.csm_node_id=a.csm_node_id||y++;a.csm_sensitive_info=e;if("SCRIPT"!=a.nodeName)for(var d=0;d<f;d++)k(a.childNodes[d],e?b:o,c);k(a,q,c)},function(a,c){var f=a.childNodes.length,e=a.parentElement||{};a.csm_blacklist_parent=e.csm_blacklist_parent||
e;if("SCRIPT"!=a.nodeName)for(e=0;e<f;e++)k(a.childNodes[e],b,c);k(a.csm_blacklist_parent,q,c)},function(a,b){var c=a.csm_blacklist_parent,f=a.csm_sensitive_info;if(c)return k(c,q,b);l.log({t:b,s:D++,k:"snpm",n:a.nodeName,id:a.csm_node_id,pid:a.parentElement?a.parentElement.csm_node_id:void 0,sid:a.nextSibling?a.nextSibling.csm_node_id:void 0,attr:i(a),txt:n(a,f),bb:e(a,f)},{glog:1})},function(a,b){var c=a.csm_blacklist_parent;if(c)return k(c,q,b);l.log({k:"snpd",s:D++,id:a.csm_node_id,t:b},{glog:1})}];
return{on:function(a){var m;d=a.bs||200;r=a.ts;"function"==typeof uex&&uex("ld","csmCELLSSpanshot",{wb:1});m=(a=a.re)||c.getElementsByTagName("html")[0],a=m;k(a,o,r());h();u=new MutationObserver(p);u.observe(a,{attributes:!0,childList:!0,characterData:!0,subtree:!0})},off:function(){u&&u.disconnect()},ready:function(){return a.ue_cel&&l.log},reset:function(){x=[];w=[]}}}(),p.ue_cel&&p.ue_rpl&&p.ue_cel.registerModule("replay module",p.ue_rpl))})(window,document,ue_csm,ue_cel);
(function(a,c){a.ue_kpm||a.ue.isBF||(a.ue_kpm=function(){function p(a){0===i.length&&j.log();i.push({w:n?a.which:0,t:k()})}function l(){if(i.length){for(var a=e,c=n?"kpmf":"kpm",j=[],k=0;k<i.length;k++){var b=i[k],b=b.t<<8|b.w;g?j.push(b-g):j.push(b);g=b}a({k:c,v:j});i=[];g=0}}var k,j=a.ue,e=a.ue_cel.log,n=!!a.ue_skc,i=[],g=0;return{on:function(a){k=a.ts;j.attach&&j.attach("keydown",p,c);if(j.onflush)j.onflush(l)},off:function(){j.detach&&j.detach("keydown",p,c)},ready:function(){return a.ue_cel&&
a.ue_cel.log},reset:function(){}}}(),a.ue_cel&&a.ue_cel.registerModule("lambada module",a.ue_kpm))})(ue_csm,document);