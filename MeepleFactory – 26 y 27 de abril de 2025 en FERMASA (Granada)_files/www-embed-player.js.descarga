(function(){'use strict';var p;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
var ba=typeof Object.defineProperties=="function"?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
function ca(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var da=ca(this);function u(a,b){if(b)a:{var c=da;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&b!=null&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
u("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.h=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.h};
var d="jscomp_symbol_"+(Math.random()*1E9>>>0)+"_",e=0;return b});
u("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var d=da[b[c]];typeof d==="function"&&typeof d.prototype[a]!="function"&&ba(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ea(aa(this))}})}return a});
function ea(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
var fa=typeof Object.create=="function"?Object.create:function(a){function b(){}
b.prototype=a;return new b},ia=function(){function a(){function c(){}
new c;Reflect.construct(c,[],function(){});
return new c instanceof c}
if(typeof Reflect!="undefined"&&Reflect.construct){if(a())return Reflect.construct;var b=Reflect.construct;return function(c,d,e){c=b(c,d);e&&Reflect.setPrototypeOf(c,e.prototype);return c}}return function(c,d,e){e===void 0&&(e=c);
e=fa(e.prototype||Object.prototype);return Function.prototype.apply.call(c,e,d)||e}}(),ja;
if(typeof Object.setPrototypeOf=="function")ja=Object.setPrototypeOf;else{var ka;a:{var la={a:!0},ma={};try{ma.__proto__=la;ka=ma.a;break a}catch(a){}ka=!1}ja=ka?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var na=ja;
function v(a,b){a.prototype=fa(b.prototype);a.prototype.constructor=a;if(na)na(a,b);else for(var c in b)if(c!="prototype")if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.Aa=b.prototype}
function y(a){var b=typeof Symbol!="undefined"&&Symbol.iterator&&a[Symbol.iterator];if(b)return b.call(a);if(typeof a.length=="number")return{next:aa(a)};throw Error(String(a)+" is not an iterable or ArrayLike");}
function A(a){if(!(a instanceof Array)){a=y(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a}
function oa(a){return pa(a,a)}
function pa(a,b){a.raw=b;Object.freeze&&(Object.freeze(a),Object.freeze(b));return a}
function qa(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var ra=typeof Object.assign=="function"?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)qa(d,e)&&(a[e]=d[e])}return a};
u("Object.assign",function(a){return a||ra});
function sa(){this.A=!1;this.u=null;this.i=void 0;this.h=1;this.o=this.H=0;this.P=this.j=null}
function ta(a){if(a.A)throw new TypeError("Generator is already running");a.A=!0}
sa.prototype.G=function(a){this.i=a};
function ua(a,b){a.j={exception:b,zd:!0};a.h=a.H||a.o}
sa.prototype.return=function(a){this.j={return:a};this.h=this.o};
sa.prototype.yield=function(a,b){this.h=b;return{value:a}};
sa.prototype.B=function(a){this.h=a};
function va(a,b,c){a.H=b;c!=void 0&&(a.o=c)}
function wa(a,b){a.h=b;a.H=0}
function xa(a){a.H=0;var b=a.j.exception;a.j=null;return b}
function ya(a){var b=a.P.splice(0)[0];(b=a.j=a.j||b)?b.zd?a.h=a.H||a.o:b.B!=void 0&&a.o<b.B?(a.h=b.B,a.j=null):a.h=a.o:a.h=0}
function za(a){this.h=new sa;this.i=a}
function Aa(a,b){ta(a.h);var c=a.h.u;if(c)return Ba(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.h.return);
a.h.return(b);return Ca(a)}
function Ba(a,b,c,d){try{var e=b.call(a.h.u,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.h.A=!1,e;var f=e.value}catch(g){return a.h.u=null,ua(a.h,g),Ca(a)}a.h.u=null;d.call(a.h,f);return Ca(a)}
function Ca(a){for(;a.h.h;)try{var b=a.i(a.h);if(b)return a.h.A=!1,{value:b.value,done:!1}}catch(c){a.h.i=void 0,ua(a.h,c)}a.h.A=!1;if(a.h.j){b=a.h.j;a.h.j=null;if(b.zd)throw b.exception;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function Da(a){this.next=function(b){ta(a.h);a.h.u?b=Ba(a,a.h.u.next,b,a.h.G):(a.h.G(b),b=Ca(a));return b};
this.throw=function(b){ta(a.h);a.h.u?b=Ba(a,a.h.u["throw"],b,a.h.G):(ua(a.h,b),b=Ca(a));return b};
this.return=function(b){return Aa(a,b)};
this[Symbol.iterator]=function(){return this}}
function Ea(a){function b(d){return a.next(d)}
function c(d){return a.throw(d)}
return new Promise(function(d,e){function f(g){g.done?d(g.value):Promise.resolve(g.value).then(b,c).then(f,e)}
f(a.next())})}
function B(a){return Ea(new Da(new za(a)))}
function C(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}
u("globalThis",function(a){return a||da});
u("Reflect",function(a){return a?a:{}});
u("Reflect.construct",function(){return ia});
u("Reflect.setPrototypeOf",function(a){return a?a:na?function(b,c){try{return na(b,c),!0}catch(d){return!1}}:null});
u("Promise",function(a){function b(g){this.X=0;this.bb=void 0;this.h=[];this.u=!1;var h=this.i();try{g(h.resolve,h.reject)}catch(k){h.reject(k)}}
function c(){this.h=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.i=function(g){if(this.h==null){this.h=[];var h=this;this.j(function(){h.u()})}this.h.push(g)};
var e=da.setTimeout;c.prototype.j=function(g){e(g,0)};
c.prototype.u=function(){for(;this.h&&this.h.length;){var g=this.h;this.h=[];for(var h=0;h<g.length;++h){var k=g[h];g[h]=null;try{k()}catch(l){this.o(l)}}}this.h=null};
c.prototype.o=function(g){this.j(function(){throw g;})};
b.prototype.i=function(){function g(l){return function(m){k||(k=!0,l.call(h,m))}}
var h=this,k=!1;return{resolve:g(this.U),reject:g(this.j)}};
b.prototype.U=function(g){if(g===this)this.j(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.Z(g);else{a:switch(typeof g){case "object":var h=g!=null;break a;case "function":h=!0;break a;default:h=!1}h?this.P(g):this.o(g)}};
b.prototype.P=function(g){var h=void 0;try{h=g.then}catch(k){this.j(k);return}typeof h=="function"?this.ha(h,g):this.o(g)};
b.prototype.j=function(g){this.H(2,g)};
b.prototype.o=function(g){this.H(1,g)};
b.prototype.H=function(g,h){if(this.X!=0)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.X);this.X=g;this.bb=h;this.X===2&&this.Y();this.A()};
b.prototype.Y=function(){var g=this;e(function(){if(g.G()){var h=da.console;typeof h!=="undefined"&&h.error(g.bb)}},1)};
b.prototype.G=function(){if(this.u)return!1;var g=da.CustomEvent,h=da.Event,k=da.dispatchEvent;if(typeof k==="undefined")return!0;typeof g==="function"?g=new g("unhandledrejection",{cancelable:!0}):typeof h==="function"?g=new h("unhandledrejection",{cancelable:!0}):(g=da.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.bb;return k(g)};
b.prototype.A=function(){if(this.h!=null){for(var g=0;g<this.h.length;++g)f.i(this.h[g]);this.h=null}};
var f=new c;b.prototype.Z=function(g){var h=this.i();g.kc(h.resolve,h.reject)};
b.prototype.ha=function(g,h){var k=this.i();try{g.call(h,k.resolve,k.reject)}catch(l){k.reject(l)}};
b.prototype.then=function(g,h){function k(r,t){return typeof r=="function"?function(w){try{l(r(w))}catch(x){m(x)}}:t}
var l,m,n=new b(function(r,t){l=r;m=t});
this.kc(k(g,l),k(h,m));return n};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.kc=function(g,h){function k(){switch(l.X){case 1:g(l.bb);break;case 2:h(l.bb);break;default:throw Error("Unexpected state: "+l.X);}}
var l=this;this.h==null?f.i(k):this.h.push(k);this.u=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,k){k(g)})};
b.race=function(g){return new b(function(h,k){for(var l=y(g),m=l.next();!m.done;m=l.next())d(m.value).kc(h,k)})};
b.all=function(g){var h=y(g),k=h.next();return k.done?d([]):new b(function(l,m){function n(w){return function(x){r[w]=x;t--;t==0&&l(r)}}
var r=[],t=0;do r.push(void 0),t++,d(k.value).kc(n(r.length-1),m),k=h.next();while(!k.done)})};
return b});
u("Object.setPrototypeOf",function(a){return a||na});
u("Symbol.dispose",function(a){return a?a:Symbol("Symbol.dispose")});
u("WeakMap",function(a){function b(k){this.h=(h+=Math.random()+1).toString();if(k){k=y(k);for(var l;!(l=k.next()).done;)l=l.value,this.set(l[0],l[1])}}
function c(){}
function d(k){var l=typeof k;return l==="object"&&k!==null||l==="function"}
function e(k){if(!qa(k,g)){var l=new c;ba(k,g,{value:l})}}
function f(k){var l=Object[k];l&&(Object[k]=function(m){if(m instanceof c)return m;Object.isExtensible(m)&&e(m);return l(m)})}
if(function(){if(!a||!Object.seal)return!1;try{var k=Object.seal({}),l=Object.seal({}),m=new a([[k,2],[l,3]]);if(m.get(k)!=2||m.get(l)!=3)return!1;m.delete(k);m.set(l,4);return!m.has(k)&&m.get(l)==4}catch(n){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(k,l){if(!d(k))throw Error("Invalid WeakMap key");e(k);if(!qa(k,g))throw Error("WeakMap key fail: "+k);k[g][this.h]=l;return this};
b.prototype.get=function(k){return d(k)&&qa(k,g)?k[g][this.h]:void 0};
b.prototype.has=function(k){return d(k)&&qa(k,g)&&qa(k[g],this.h)};
b.prototype.delete=function(k){return d(k)&&qa(k,g)&&qa(k[g],this.h)?delete k[g][this.h]:!1};
return b});
u("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,k){var l=h[1];return ea(function(){if(l){for(;l.head!=h[1];)l=l.previous;for(;l.next!=l.head;)return l=l.next,{done:!1,value:k(l)};l=null}return{done:!0,value:void 0}})}
function d(h,k){var l=k&&typeof k;l=="object"||l=="function"?f.has(k)?l=f.get(k):(l=""+ ++g,f.set(k,l)):l="p_"+k;var m=h[0][l];if(m&&qa(h[0],l))for(h=0;h<m.length;h++){var n=m[h];if(k!==k&&n.key!==n.key||k===n.key)return{id:l,list:m,index:h,entry:n}}return{id:l,list:m,index:-1,entry:void 0}}
function e(h){this[0]={};this[1]=b();this.size=0;if(h){h=y(h);for(var k;!(k=h.next()).done;)k=k.value,this.set(k[0],k[1])}}
if(function(){if(!a||typeof a!="function"||!a.prototype.entries||typeof Object.seal!="function")return!1;try{var h=Object.seal({x:4}),k=new a(y([[h,"s"]]));if(k.get(h)!="s"||k.size!=1||k.get({x:4})||k.set({x:4},"t")!=k||k.size!=2)return!1;var l=k.entries(),m=l.next();if(m.done||m.value[0]!=h||m.value[1]!="s")return!1;m=l.next();return m.done||m.value[0].x!=4||m.value[1]!="t"||!l.next().done?!1:!0}catch(n){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,k){h=h===0?0:h;var l=d(this,h);l.list||(l.list=this[0][l.id]=[]);l.entry?l.entry.value=k:(l.entry={next:this[1],previous:this[1].previous,head:this[1],key:h,value:k},l.list.push(l.entry),this[1].previous.next=l.entry,this[1].previous=l.entry,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.entry&&h.list?(h.list.splice(h.index,1),h.list.length||delete this[0][h.id],h.entry.previous.next=h.entry.next,h.entry.next.previous=h.entry.previous,h.entry.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this[0]={};this[1]=this[1].previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).entry};
e.prototype.get=function(h){return(h=d(this,h).entry)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,k){for(var l=this.entries(),m;!(m=l.next()).done;)m=m.value,h.call(k,m[1],m[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
u("Set",function(a){function b(c){this.h=new Map;if(c){c=y(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.h.size}
if(function(){if(!a||typeof a!="function"||!a.prototype.entries||typeof Object.seal!="function")return!1;try{var c=Object.seal({x:4}),d=new a(y([c]));if(!d.has(c)||d.size!=1||d.add(c)!=d||d.size!=1||d.add({x:4})!=d||d.size!=2)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||f.value[0].x!=4||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=c===0?0:c;this.h.set(c,c);this.size=this.h.size;return this};
b.prototype.delete=function(c){c=this.h.delete(c);this.size=this.h.size;return c};
b.prototype.clear=function(){this.h.clear();this.size=0};
b.prototype.has=function(c){return this.h.has(c)};
b.prototype.entries=function(){return this.h.entries()};
b.prototype.values=function(){return this.h.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.h.forEach(function(f){return c.call(d,f,f,e)})};
return b});
function Fa(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
u("Array.prototype.entries",function(a){return a?a:function(){return Fa(this,function(b,c){return[b,c]})}});
u("Array.prototype.keys",function(a){return a?a:function(){return Fa(this,function(b){return b})}});
function Ga(a,b,c){if(a==null)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
u("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Ga(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
u("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Ga(this,b,"endsWith");b+="";c===void 0&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;e>0&&c>0;)if(d[--c]!=b[--e])return!1;return e<=0}});
u("Number.isFinite",function(a){return a?a:function(b){return typeof b!=="number"?!1:!isNaN(b)&&b!==Infinity&&b!==-Infinity}});
u("Array.prototype.find",function(a){return a?a:function(b,c){a:{var d=this;d instanceof String&&(d=String(d));for(var e=d.length,f=0;f<e;f++){var g=d[f];if(b.call(c,g,f,d)){b=g;break a}}b=void 0}return b}});
u("Object.values",function(a){return a?a:function(b){var c=[],d;for(d in b)qa(b,d)&&c.push(b[d]);return c}});
u("Object.is",function(a){return a?a:function(b,c){return b===c?b!==0||1/b===1/c:b!==b&&c!==c}});
u("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(c<0&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
u("String.prototype.includes",function(a){return a?a:function(b,c){return Ga(this,b,"includes").indexOf(b,c||0)!==-1}});
u("Array.from",function(a){return a?a:function(b,c,d){c=c!=null?c:function(h){return h};
var e=[],f=typeof Symbol!="undefined"&&Symbol.iterator&&b[Symbol.iterator];if(typeof f=="function"){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
u("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)qa(b,d)&&c.push([d,b[d]]);return c}});
u("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
u("Number.MIN_SAFE_INTEGER",function(){return-9007199254740991});
u("Number.isInteger",function(a){return a?a:function(b){return Number.isFinite(b)?b===Math.floor(b):!1}});
u("Number.isSafeInteger",function(a){return a?a:function(b){return Number.isInteger(b)&&Math.abs(b)<=Number.MAX_SAFE_INTEGER}});
u("Math.trunc",function(a){return a?a:function(b){b=Number(b);if(isNaN(b)||b===Infinity||b===-Infinity||b===0)return b;var c=Math.floor(Math.abs(b));return b<0?-c:c}});
u("Number.isNaN",function(a){return a?a:function(b){return typeof b==="number"&&isNaN(b)}});
u("Array.prototype.values",function(a){return a?a:function(){return Fa(this,function(b,c){return c})}});/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var Ha=Ha||{},D=this||self;function E(a,b,c){a=a.split(".");c=c||D;for(var d;a.length&&(d=a.shift());)a.length||b===void 0?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function Ka(a){var b=F("CLOSURE_FLAGS");a=b&&b[a];return a!=null?a:!1}
function F(a,b){a=a.split(".");b=b||D;for(var c=0;c<a.length;c++)if(b=b[a[c]],b==null)return null;return b}
function La(a){var b=typeof a;return b!="object"?b:a?Array.isArray(a)?"array":b:"null"}
function Ma(a){var b=La(a);return b=="array"||b=="object"&&typeof a.length=="number"}
function Oa(a){var b=typeof a;return b=="object"&&a!=null||b=="function"}
function Pa(a){return Object.prototype.hasOwnProperty.call(a,Qa)&&a[Qa]||(a[Qa]=++Ra)}
var Qa="closure_uid_"+(Math.random()*1E9>>>0),Ra=0;function Sa(a,b,c){return a.call.apply(a.bind,arguments)}
function Ta(a,b,c){if(!a)throw Error();if(arguments.length>2){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Ua(a,b,c){Ua=Function.prototype.bind&&Function.prototype.bind.toString().indexOf("native code")!=-1?Sa:Ta;return Ua.apply(null,arguments)}
function Va(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}}
function Ya(){return Date.now()}
function Za(a){return a}
function $a(a,b){function c(){}
c.prototype=b.prototype;a.Aa=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.base=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
;function ab(a,b){return(a=(new RegExp("[^#]*[?&]"+b+"=([^&#]*)")).exec(a))?a[1]:null}
function bb(a){var b=C.apply(1,arguments).filter(Boolean).join("&");if(!b)return a;var c=a.match(/[?&]adurl=/);return c?a.slice(0,c.index+1)+b+"&"+a.slice(c.index+1):a+(a.indexOf("?")<0?"?":"&")+b}
function cb(a,b){return b?"&"+a+"="+encodeURIComponent(b):""}
function db(a){var b=a.url;a=a.li;this.h=b;this.j=a;this.i=(new Date).getTime()-17040672E5}
function eb(a){a=a.j;if(!a)return"";var b=cb("uap",a.platform)+cb("uapv",a.platformVersion)+cb("uafv",a.uaFullVersion)+cb("uaa",a.architecture)+cb("uam",a.model)+cb("uab",a.bitness);a.fullVersionList&&(b+="&uafvl="+encodeURIComponent(a.fullVersionList.map(function(c){return encodeURIComponent(c.brand)+";"+encodeURIComponent(c.version)}).join("|")));
a.wow64!=null&&(b+="&uaw="+Number(a.wow64));return b.slice(1)}
;function fb(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,fb);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));b!==void 0&&(this.cause=b)}
$a(fb,Error);fb.prototype.name="CustomError";function gb(a){return a}
;var hb=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};/*

 Copyright Google LLC
 SPDX-License-Identifier: Apache-2.0
*/
var ib=globalThis.trustedTypes,jb;function kb(){var a=null;if(!ib)return a;try{var b=function(c){return c};
a=ib.createPolicy("goog#html",{createHTML:b,createScript:b,createScriptURL:b})}catch(c){}return a}
function lb(){jb===void 0&&(jb=kb());return jb}
;function nb(a){this.h=a}
nb.prototype.toString=function(){return this.h+""};
function ob(a){var b=lb();a=b?b.createScriptURL(a):a;return new nb(a)}
function pb(a){if(a instanceof nb)return a.h;throw Error("");}
;var qb=oa([""]),rb=pa(["\x00"],["\\0"]),sb=pa(["\n"],["\\n"]),tb=pa(["\x00"],["\\u0000"]);function ub(a){return a.toString().indexOf("`")===-1}
ub(function(a){return a(qb)})||ub(function(a){return a(rb)})||ub(function(a){return a(sb)})||ub(function(a){return a(tb)});function vb(a){this.h=a}
vb.prototype.toString=function(){return this.h};
var wb=new vb("about:invalid#zClosurez");function xb(a){this.Ge=a}
function yb(a){return new xb(function(b){return b.substr(0,a.length+1).toLowerCase()===a+":"})}
var zb=[yb("data"),yb("http"),yb("https"),yb("mailto"),yb("ftp"),new xb(function(a){return/^[^:]*([/?#]|$)/.test(a)})],Ab=/^\s*(?!javascript:)(?:[\w+.-]+:|[^:/?#]*(?:[/?#]|$))/i;
function Bb(a){if(a instanceof vb)if(a instanceof vb)a=a.h;else throw Error("");else a=Ab.test(a)?a:void 0;return a}
;function Db(a,b){b=Bb(b);b!==void 0&&(a.href=b)}
;function Eb(a,b){throw Error(b===void 0?"unexpected value "+a+"!":b);}
;function Fb(a){this.h=a}
Fb.prototype.toString=function(){return this.h+""};function Gb(a){a=a===void 0?document:a;var b,c;a=(c=(b=a).querySelector)==null?void 0:c.call(b,"script[nonce]");return a==null?"":a.nonce||a.getAttribute("nonce")||""}
;function Hb(a){this.h=a}
Hb.prototype.toString=function(){return this.h+""};
function Ib(a){var b=lb();a=b?b.createScript(a):a;return new Hb(a)}
function Jb(a){if(a instanceof Hb)return a.h;throw Error("");}
;function Kb(a){var b=Gb(a.ownerDocument);b&&a.setAttribute("nonce",b)}
function Lb(a,b){a.src=pb(b);Kb(a)}
;function Mb(){this.h=Nb[0].toLowerCase()}
Mb.prototype.toString=function(){return this.h};function Ob(a){var b="true".toString(),c=[new Mb];if(c.length===0)throw Error("");if(c.map(function(d){if(d instanceof Mb)d=d.h;else throw Error("");return d}).every(function(d){return"data-loaded".indexOf(d)!==0}))throw Error('Attribute "data-loaded" does not match any of the allowed prefixes.');
a.setAttribute("data-loaded",b)}
;var Pb="alternate author bookmark canonical cite help icon license modulepreload next prefetch dns-prefetch prerender preconnect preload prev search subresource".split(" ");function Qb(a,b){if(b instanceof nb)a.href=pb(b).toString(),a.rel="stylesheet";else{if(Pb.indexOf("stylesheet")===-1)throw Error('TrustedResourceUrl href attribute required with rel="stylesheet"');b=Bb(b);b!==void 0&&(a.href=b,a.rel="stylesheet")}}
;var Sb=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if(typeof a==="string")return typeof b!=="string"||b.length!=1?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},Tb=Array.prototype.forEach?function(a,b){Array.prototype.forEach.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=typeof a==="string"?a.split(""):a,e=0;e<c;e++)e in d&&b.call(void 0,d[e],e,a)},Ub=Array.prototype.filter?function(a,b){return Array.prototype.filter.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=[],e=0,f=typeof a==="string"?a.split(""):a,g=0;g<c;g++)if(g in f){var h=f[g];
b.call(void 0,h,g,a)&&(d[e++]=h)}return d},Vb=Array.prototype.map?function(a,b){return Array.prototype.map.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=Array(c),e=typeof a==="string"?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));
return d},Wb=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
Tb(a,function(e,f){d=b.call(void 0,d,e,f,a)});
return d};
function Xb(a,b){a:{for(var c=a.length,d=typeof a==="string"?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return b<0?null:typeof a==="string"?a.charAt(b):a[b]}
function Yb(a,b){b=Sb(a,b);var c;(c=b>=0)&&Array.prototype.splice.call(a,b,1);return c}
function Zb(a){var b=a.length;if(b>0){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function $b(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Ma(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function ac(a,b){a.__closure__error__context__984382||(a.__closure__error__context__984382={});a.__closure__error__context__984382.severity=b}
;function bc(a){var b=F("window.location.href");a==null&&(a='Unknown Error of type "null/undefined"');if(typeof a==="string")return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||D.$googDebugFname||b}catch(g){e="Not available",c=!0}b=cc(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(c==
null){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,dc[c])c=dc[c];else{c=String(c);if(!dc[c]){var f=/function\s+([^\(]+)/m.exec(c);dc[c]=f?f[1]:"[Anonymous]"}c=dc[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";typeof a.toString==="function"&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}return{message:a.message,
name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:b}}
function cc(a,b){b||(b={});b[ec(a)]=!0;var c=a.stack||"",d=a.cause;d&&!b[ec(d)]&&(c+="\nCaused by: ",d.stack&&d.stack.indexOf(d.toString())==0||(c+=typeof d==="string"?d:d.message+"\n"),c+=cc(d,b));a=a.errors;if(Array.isArray(a)){d=1;var e;for(e=0;e<a.length&&!(d>4);e++)b[ec(a[e])]||(c+="\nInner error "+d++ +": ",a[e].stack&&a[e].stack.indexOf(a[e].toString())==0||(c+=typeof a[e]==="string"?a[e]:a[e].message+"\n"),c+=cc(a[e],b));e<a.length&&(c+="\n... "+(a.length-e)+" more inner errors")}return c}
function ec(a){var b="";typeof a.toString==="function"&&(b=""+a);return b+a.stack}
var dc={};function fc(a){return decodeURIComponent(a.replace(/\+/g," "))}
function hc(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var ic=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function jc(a){return a?decodeURI(a):a}
function kc(a){return jc(a.match(ic)[3]||null)}
function lc(a){return jc(a.match(ic)[5]||null)}
function mc(a){var b=a.match(ic);a=b[5];var c=b[6];b=b[7];var d="";a&&(d+=a);c&&(d+="?"+c);b&&(d+="#"+b);return d}
function nc(a){var b=a.indexOf("#");return b<0?a:a.slice(0,b)}
function oc(a,b){if(a){a=a.split("&");for(var c=0;c<a.length;c++){var d=a[c].indexOf("="),e=null;if(d>=0){var f=a[c].substring(0,d);e=a[c].substring(d+1)}else f=a[c];b(f,e?fc(e):"")}}}
function pc(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)pc(a,String(b[d]),c);else b!=null&&c.push(a+(b===""?"":"="+encodeURIComponent(String(b))))}
function qc(a){var b=[],c;for(c in a)pc(c,a[c],b);return b.join("&")}
function rc(a,b){b=qc(b);if(b){var c=a.indexOf("#");c<0&&(c=a.length);var d=a.indexOf("?");if(d<0||d>c){d=c;var e=""}else e=a.substring(d+1,c);a=[a.slice(0,d),e,a.slice(c)];c=a[1];a[1]=b?c?c+"&"+b:b:c;b=a[0]+(a[1]?"?"+a[1]:"")+a[2]}else b=a;return b}
function sc(a,b,c,d){for(var e=c.length;(b=a.indexOf(c,b))>=0&&b<d;){var f=a.charCodeAt(b-1);if(f==38||f==63)if(f=a.charCodeAt(b+e),!f||f==61||f==38||f==35)return b;b+=e+1}return-1}
var tc=/#|$/,uc=/[?&]($|#)/;function vc(a,b){for(var c=a.search(tc),d=0,e,f=[];(e=sc(a,d,b,c))>=0;)f.push(a.substring(d,e)),d=Math.min(a.indexOf("&",e)+1||c,c);f.push(a.slice(d));return f.join("").replace(uc,"$1")}
;function wc(){try{var a,b;return!!((a=window)==null?0:(b=a.top)==null?0:b.location.href)&&!1}catch(c){return!0}}
;function xc(a){a&&typeof a.dispose=="function"&&a.dispose()}
;function yc(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];Ma(d)?yc.apply(null,d):xc(d)}}
;function I(){this.ea=this.ea;this.H=this.H}
I.prototype.ea=!1;I.prototype.dispose=function(){this.ea||(this.ea=!0,this.ba())};
I.prototype[Symbol.dispose]=function(){this.dispose()};
function zc(a,b){a.addOnDisposeCallback(Va(xc,b))}
I.prototype.addOnDisposeCallback=function(a,b){this.ea?b!==void 0?a.call(b):a():(this.H||(this.H=[]),b&&(a=a.bind(b)),this.H.push(a))};
I.prototype.ba=function(){if(this.H)for(;this.H.length;)this.H.shift()()};function Ac(){var a=Bc();a=a===void 0?"bevasrsg":a;return new Promise(function(b){var c=window===window.top?window:wc()?window:window.top,d=c[a],e;((e=d)==null?0:e.bevasrs)?b(new Cc(d.bevasrs)):(d||(d={},d=(d.nqfbel=[],d),c[a]=d),d.nqfbel.push(function(f){b(new Cc(f))}))})}
function Cc(a){I.call(this);var b=this;this.vm=a;this.i="keydown keypress keyup input focusin focusout select copy cut paste change click dblclick auxclick pointerover pointerdown pointerup pointermove pointerout dragenter dragleave drag dragend mouseover mousedown mouseup mousemove mouseout touchstart touchend touchmove wheel".split(" ");this.h=void 0;this.cd=this.vm.p;this.j=this.o.bind(this);this.addOnDisposeCallback(function(){return void Dc(b)})}
v(Cc,I);Cc.prototype.snapshot=function(a){return this.vm.s(Object.assign({},a.wb&&{c:a.wb},a.ed&&{s:a.ed},a.gd!==void 0&&{p:a.gd}))};
Cc.prototype.o=function(a){this.vm.e(a)};
function Dc(a){a.h!==void 0&&(a.i.forEach(function(b){var c;(c=a.h)==null||c.removeEventListener(b,a.j)}),a.h=void 0)}
;function Ec(a){var b=b===void 0?49:b;var c=[];Fc(a,Gc,6).forEach(function(d){Hc(d,2)<=b&&c.push(Hc(d,1))});
return c}
function Ic(a){var b=b===void 0?49:b;var c=[];Fc(a,Gc,6).forEach(function(d){Hc(d,2)>b&&c.push(Hc(d,1))});
return c}
;var Jc;function Kc(){I.apply(this,arguments);this.j=1;this[Jc]=this.dispose}
v(Kc,I);Kc.prototype.share=function(){if(this.ea)throw Error("E:AD");this.j++;return this};
Kc.prototype.dispose=function(){--this.j||I.prototype.dispose.call(this)};
Jc=Symbol.dispose;function Lc(a){return{fieldType:2,fieldName:a}}
function Mc(a){return{fieldType:3,fieldName:a}}
;function Nc(a){this.h=a;a.Jc("/client_streamz/bg/frs",Mc("mk"))}
Nc.prototype.record=function(a,b){this.h.record("/client_streamz/bg/frs",a,b)};
function Oc(a){this.h=a;a.Jc("/client_streamz/bg/wrl",Mc("mn"),Lc("ac"),Lc("sc"),Mc("rk"),Mc("mk"))}
Oc.prototype.record=function(a,b,c,d,e,f){this.h.record("/client_streamz/bg/wrl",a,b,c,d,e,f)};
function Pc(a){this.h=a;a.Nb("/client_streamz/bg/ec",Mc("en"),Mc("mk"))}
Pc.prototype.kb=function(a,b){this.h.Kb("/client_streamz/bg/ec",a,b)};
function Qc(a){this.h=a;a.Jc("/client_streamz/bg/el",Mc("en"),Mc("mk"))}
Qc.prototype.record=function(a,b,c){this.h.record("/client_streamz/bg/el",a,b,c)};
function Rc(a){this.h=a;a.Nb("/client_streamz/bg/cec",Lc("ec"),Mc("mk"))}
Rc.prototype.kb=function(a,b){this.h.Kb("/client_streamz/bg/cec",a,b)};
function Sc(a){this.h=a;a.Nb("/client_streamz/bg/po/csc",Lc("cs"),Mc("mk"))}
Sc.prototype.kb=function(a,b){this.h.Kb("/client_streamz/bg/po/csc",a,b)};
function Tc(a){this.h=a;a.Nb("/client_streamz/bg/po/ctav",Mc("av"),Mc("mk"))}
Tc.prototype.kb=function(a,b){this.h.Kb("/client_streamz/bg/po/ctav",a,b)};
function Uc(a){this.h=a;a.Nb("/client_streamz/bg/po/cwsc",Mc("su"),Mc("mk"))}
Uc.prototype.kb=function(a,b){this.h.Kb("/client_streamz/bg/po/cwsc",a,b)};function Vc(a){D.setTimeout(function(){throw a;},0)}
;var Wc=Ka(610401301),Xc=Ka(513659523),Yc=Ka(568333945),Zc=Ka(1981196515);function $c(){var a=D.navigator;return a&&(a=a.userAgent)?a:""}
var ad,bd=D.navigator;ad=bd?bd.userAgentData||null:null;function cd(a){if(!Wc||!ad)return!1;for(var b=0;b<ad.brands.length;b++){var c=ad.brands[b].brand;if(c&&c.indexOf(a)!=-1)return!0}return!1}
function J(a){return $c().indexOf(a)!=-1}
;function dd(){return Wc?!!ad&&ad.brands.length>0:!1}
function ed(){return dd()?!1:J("Opera")}
function fd(){return J("Firefox")||J("FxiOS")}
function gd(){return dd()?cd("Chromium"):(J("Chrome")||J("CriOS"))&&!(dd()?0:J("Edge"))||J("Silk")}
;function hd(){return Wc?!!ad&&!!ad.platform:!1}
function id(){return J("iPhone")&&!J("iPod")&&!J("iPad")}
;function jd(a){jd[" "](a);return a}
jd[" "]=function(){};var kd=ed(),ld=dd()?!1:J("Trident")||J("MSIE"),md=J("Edge"),nd=J("Gecko")&&!($c().toLowerCase().indexOf("webkit")!=-1&&!J("Edge"))&&!(J("Trident")||J("MSIE"))&&!J("Edge"),od=$c().toLowerCase().indexOf("webkit")!=-1&&!J("Edge");od&&J("Mobile");hd()||J("Macintosh");hd()||J("Windows");(hd()?ad.platform==="Linux":J("Linux"))||hd()||J("CrOS");var pd=hd()?ad.platform==="Android":J("Android");id();J("iPad");J("iPod");id()||J("iPad")||J("iPod");$c().toLowerCase().indexOf("kaios");fd();var qd=id()||J("iPod"),rd=J("iPad");!J("Android")||gd()||fd()||ed()||J("Silk");gd();var sd=J("Safari")&&!(gd()||(dd()?0:J("Coast"))||ed()||(dd()?0:J("Edge"))||(dd()?cd("Microsoft Edge"):J("Edg/"))||(dd()?cd("Opera"):J("OPR"))||fd()||J("Silk")||J("Android"))&&!(id()||J("iPad")||J("iPod"));var td={},ud=null;function vd(a,b){Ma(a);b===void 0&&(b=0);wd();b=td[b];for(var c=Array(Math.floor(a.length/3)),d=b[64]||"",e=0,f=0;e<a.length-2;e+=3){var g=a[e],h=a[e+1],k=a[e+2],l=b[g>>2];g=b[(g&3)<<4|h>>4];h=b[(h&15)<<2|k>>6];k=b[k&63];c[f++]=""+l+g+h+k}l=0;k=d;switch(a.length-e){case 2:l=a[e+1],k=b[(l&15)<<2]||d;case 1:a=a[e],c[f]=""+b[a>>2]+b[(a&3)<<4|l>>4]+k+d}return c.join("")}
function xd(a){var b=a.length,c=b*3/4;c%3?c=Math.floor(c):"=.".indexOf(a[b-1])!=-1&&(c="=.".indexOf(a[b-2])!=-1?c-2:c-1);var d=new Uint8Array(c),e=0;yd(a,function(f){d[e++]=f});
return e!==c?d.subarray(0,e):d}
function yd(a,b){function c(k){for(;d<a.length;){var l=a.charAt(d++),m=ud[l];if(m!=null)return m;if(!/^[\s\xa0]*$/.test(l))throw Error("Unknown base64 encoding at char: "+l);}return k}
wd();for(var d=0;;){var e=c(-1),f=c(0),g=c(64),h=c(64);if(h===64&&e===-1)break;b(e<<2|f>>4);g!=64&&(b(f<<4&240|g>>2),h!=64&&b(g<<6&192|h))}}
function wd(){if(!ud){ud={};for(var a="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),b=["+/=","+/","-_=","-_.","-_"],c=0;c<5;c++){var d=a.concat(b[c].split(""));td[c]=d;for(var e=0;e<d.length;e++){var f=d[e];ud[f]===void 0&&(ud[f]=e)}}}}
;var zd=typeof Uint8Array!=="undefined",Ad=!ld&&typeof btoa==="function",Bd=/[-_.]/g,Cd={"-":"+",_:"/",".":"="};function Dd(a){return Cd[a]||""}
var Ed={};function Fd(a,b){Gd(b);this.h=a;if(a!=null&&a.length===0)throw Error("ByteString should be constructed with non-empty values");}
Fd.prototype.sizeBytes=function(){Gd(Ed);var a=this.h;if(!(a==null||zd&&a!=null&&a instanceof Uint8Array))if(typeof a==="string")if(Ad){a=Bd.test(a)?a.replace(Bd,Dd):a;a=atob(a);for(var b=new Uint8Array(a.length),c=0;c<a.length;c++)b[c]=a.charCodeAt(c);a=b}else a=xd(a);else La(a),a=null;return(a=a==null?a:this.h=a)?a.length:0};
var Hd;function Gd(a){if(a!==Ed)throw Error("illegal external caller");}
;var Id=void 0;function Jd(a){a=Error(a);ac(a,"warning");return a}
function Kd(a,b){if(a!=null){var c;var d=(c=Id)!=null?c:Id={};c=d[a]||0;c>=b||(d[a]=c+1,a=Error(),ac(a,"incident"),Vc(a))}}
;var Ld=typeof Symbol==="function"&&typeof Symbol()==="symbol";function Md(a,b,c){return typeof Symbol==="function"&&typeof Symbol()==="symbol"?(c===void 0?0:c)&&Symbol.for&&a?Symbol.for(a):a!=null?Symbol(a):Symbol():b}
var Nd=Md("jas",void 0,!0),Od=Md(void 0,"1oa"),Pd=Md(void 0,Symbol()),Qd=Md(void 0,"0ub"),Rd=Md(void 0,"0ubs"),Sd=Md(void 0,"0actk"),Td=Md("m_m","Uh",!0),Ud=Md(void 0,"vps"),Vd=Md();Math.max.apply(Math,A(Object.values({mh:1,lh:2,kh:4,ph:8,rh:16,nh:32,Of:64,ih:128,Tf:256,qh:512,Uf:1024,jh:2048,oh:4096})));var Wd={Fe:{value:0,configurable:!0,writable:!0,enumerable:!1}},Xd=Object.defineProperties,K=Ld?Nd:"Fe",Yd,Zd=[];$d(Zd,7);Yd=Object.freeze(Zd);function ae(a,b){Ld||K in a||Xd(a,Wd);a[K]|=b}
function $d(a,b){Ld||K in a||Xd(a,Wd);a[K]=b}
;function be(){return typeof BigInt==="function"}
;var ce={};function de(a,b){if(b===void 0){if(b=a.h!==ee)b=!!(2&(a.F[K]|0));return b}return!!(2&b)&&a.h!==ee}
var ee={},fe=Object.freeze({}),ge={};function he(a){a.Ph=!0;return a}
;var ie=he(function(a){return typeof a==="number"}),je=he(function(a){return typeof a==="string"}),ke=he(function(a){return typeof a==="boolean"});
function le(){var a=me;return he(function(b){for(var c in a)if(b===a[c]&&!/^[0-9]+$/.test(c))return!0;return!1})}
var ne=he(function(a){return a!=null&&typeof a==="object"&&typeof a.then==="function"});var oe=typeof D.BigInt==="function"&&typeof D.BigInt(0)==="bigint";function pe(a){var b=a;if(je(b)){if(!/^\s*(?:-?[1-9]\d*|0)?\s*$/.test(b))throw Error(String(b));}else if(ie(b)&&!Number.isSafeInteger(b))throw Error(String(b));return oe?BigInt(a):a=ke(a)?a?"1":"0":je(a)?a.trim()||"0":String(a)}
var ve=he(function(a){return oe?a>=qe&&a<=re:a[0]==="-"?se(a,te):se(a,ue)}),te=Number.MIN_SAFE_INTEGER.toString(),qe=oe?BigInt(Number.MIN_SAFE_INTEGER):void 0,ue=Number.MAX_SAFE_INTEGER.toString(),re=oe?BigInt(Number.MAX_SAFE_INTEGER):void 0;
function se(a,b){if(a.length>b.length)return!1;if(a.length<b.length||a===b)return!0;for(var c=0;c<a.length;c++){var d=a[c],e=b[c];if(d>e)return!1;if(d<e)return!0}}
;var we=0,xe=0;function ye(a){var b=a>>>0;we=b;xe=(a-b)/4294967296>>>0}
function ze(a){if(a<0){ye(0-a);var b=y(Ae(we,xe));a=b.next().value;b=b.next().value;we=a>>>0;xe=b>>>0}else ye(a)}
function Be(a,b){b>>>=0;a>>>=0;if(b<=2097151)var c=""+(4294967296*b+a);else be()?c=""+(BigInt(b)<<BigInt(32)|BigInt(a)):(c=(a>>>24|b<<8)&16777215,b=b>>16&65535,a=(a&16777215)+c*6777216+b*6710656,c+=b*8147497,b*=2,a>=1E7&&(c+=a/1E7>>>0,a%=1E7),c>=1E7&&(b+=c/1E7>>>0,c%=1E7),c=b+Ce(c)+Ce(a));return c}
function Ce(a){a=String(a);return"0000000".slice(a.length)+a}
function De(){var a=we,b=xe;b&2147483648?be()?a=""+(BigInt(b|0)<<BigInt(32)|BigInt(a>>>0)):(b=y(Ae(a,b)),a=b.next().value,b=b.next().value,a="-"+Be(a,b)):a=Be(a,b);return a}
function Ae(a,b){b=~b;a?a=~a+1:b+=1;return[a,b]}
;function Ee(a){return Array.prototype.slice.call(a)}
;var Fe=typeof BigInt==="function"?BigInt.asIntN:void 0,Ge=Number.isSafeInteger,He=Number.isFinite,Ie=Math.trunc;function Je(a){return a.displayName||a.name||"unknown type name"}
function Ke(a){if(a!=null&&typeof a!=="boolean")throw Error("Expected boolean but got "+La(a)+": "+a);return a}
var Le=/^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;function Me(a){switch(typeof a){case "bigint":return!0;case "number":return He(a);case "string":return Le.test(a);default:return!1}}
function Ne(a){if(typeof a!=="number")throw Jd("int32");if(!He(a))throw Jd("int32");return a|0}
function Oe(a){return a==null?a:Ne(a)}
function Pe(a){if(a==null)return a;if(typeof a==="string"&&a)a=+a;else if(typeof a!=="number")return;return He(a)?a|0:void 0}
function Qe(a){var b=0;b=b===void 0?0:b;if(!Me(a))throw Jd("int64");var c=typeof a;switch(b){case 512:switch(c){case "string":return Re(a);case "bigint":return String(Fe(64,a));default:return Se(a)}case 1024:switch(c){case "string":return Te(a);case "bigint":return pe(Fe(64,a));default:return Ue(a)}case 0:switch(c){case "string":return Re(a);case "bigint":return pe(Fe(64,a));default:return Ve(a)}default:return Eb(b,"Unknown format requested type for int64")}}
function We(a){return a==null?a:Qe(a)}
function Xe(a){var b=a.length;return a[0]==="-"?b<20?!0:b===20&&Number(a.substring(0,7))>-922337:b<19?!0:b===19&&Number(a.substring(0,6))<922337}
function Ye(a){a.indexOf(".");if(Xe(a))return a;if(a.length<16)ze(Number(a));else if(be())a=BigInt(a),we=Number(a&BigInt(4294967295))>>>0,xe=Number(a>>BigInt(32)&BigInt(4294967295));else{var b=+(a[0]==="-");xe=we=0;for(var c=a.length,d=0+b,e=(c-b)%6+b;e<=c;d=e,e+=6)d=Number(a.slice(d,e)),xe*=1E6,we=we*1E6+d,we>=4294967296&&(xe+=Math.trunc(we/4294967296),xe>>>=0,we>>>=0);b&&(b=y(Ae(we,xe)),a=b.next().value,b=b.next().value,we=a,xe=b)}return De()}
function Ve(a){Me(a);a=Ie(a);if(!Ge(a)){ze(a);var b=we,c=xe;if(a=c&2147483648)b=~b+1>>>0,c=~c>>>0,b==0&&(c=c+1>>>0);var d=c*4294967296+(b>>>0);b=Number.isSafeInteger(d)?d:Be(b,c);a=typeof b==="number"?a?-b:b:a?"-"+b:b}return a}
function Se(a){Me(a);a=Ie(a);if(Ge(a))a=String(a);else{var b=String(a);Xe(b)?a=b:(ze(a),a=De())}return a}
function Re(a){Me(a);var b=Ie(Number(a));if(Ge(b))return String(b);b=a.indexOf(".");b!==-1&&(a=a.substring(0,b));return Ye(a)}
function Te(a){var b=Ie(Number(a));if(Ge(b))return pe(b);b=a.indexOf(".");b!==-1&&(a=a.substring(0,b));return be()?pe(Fe(64,BigInt(a))):pe(Ye(a))}
function Ue(a){return Ge(a)?pe(Ve(a)):pe(Se(a))}
function Ze(a){if(typeof a!=="string")throw Error();return a}
function $e(a){if(a!=null&&typeof a!=="string")throw Error();return a}
function af(a){return a==null||typeof a==="string"?a:void 0}
function bf(a,b){if(!(a instanceof b))throw Error("Expected instanceof "+Je(b)+" but got "+(a&&Je(a.constructor)));}
function cf(a,b,c){if(a!=null&&typeof a==="object"&&a[Td]===ce)return a;if(Array.isArray(a)){var d=a[K]|0;c=d|c&32|c&2;c!==d&&$d(a,c);return new b(a)}}
;var df={};function ef(a){return a}
;var ff={di:!0};function gf(a,b,c,d){var e=d!==void 0;d=!!d;var f=Za(Pd),g;!e&&Ld&&f&&(g=a[f])&&g.Mh(hf);f=[];var h=a.length;g=4294967295;var k=!1,l=!!(b&64),m=l?b&128?0:-1:void 0;if(!(b&1)){var n=h&&a[h-1];n!=null&&typeof n==="object"&&n.constructor===Object?(h--,g=h):n=void 0;if(l&&!(b&128)&&!e){k=!0;var r;g=((r=jf)!=null?r:ef)(g-m,m,a,n)+m}}b=void 0;for(e=0;e<h;e++)if(r=a[e],r!=null&&(r=c(r,d))!=null)if(l&&e>=g){var t=e-m,w=void 0;((w=b)!=null?w:b={})[t]=r}else f[e]=r;if(n)for(var x in n)a=n[x],a!=null&&(a=c(a,
d))!=null&&(h=+x,e=void 0,l&&!Number.isNaN(h)&&(e=h+m)<g?f[e]=a:(h=void 0,((h=b)!=null?h:b={})[x]=a));b&&(k?f.push(b):f[g]=b);return f}
function kf(a){switch(typeof a){case "number":return Number.isFinite(a)?a:""+a;case "bigint":return ve(a)?Number(a):""+a;case "boolean":return a?1:0;case "object":if(Array.isArray(a)){var b=a[K]|0;return a.length===0&&b&1?void 0:gf(a,b,kf)}if(a[Td]===ce)return lf(a);if(a instanceof Fd){b=a.h;if(b==null)a="";else if(typeof b==="string")a=b;else{if(Ad){for(var c="",d=0,e=b.length-10240;d<e;)c+=String.fromCharCode.apply(null,b.subarray(d,d+=10240));c+=String.fromCharCode.apply(null,d?b.subarray(d):b);
b=btoa(c)}else b=vd(b);a=a.h=b}return a}return}return a}
var jf;function mf(a,b){if(b){jf=b==null||b===ef||b[Ud]!==df?ef:b;try{return lf(a)}finally{jf=void 0}}return lf(a)}
function lf(a){a=a.F;return gf(a,a[K]|0,kf)}
function hf(a,b){b<500||Kd(Rd,1)}
;function L(a,b,c){var d=d===void 0?0:d;if(a==null){var e=32;c?(a=[c],e|=128):a=[];b&&(e=e&-8380417|(b&1023)<<13)}else{if(!Array.isArray(a))throw Error("narr");e=a[K]|0;2048&e&&!(2&e)&&nf();if(e&256)throw Error("farr");if(e&64)return d!==0||e&2048||$d(a,e|2048),a;if(c&&(e|=128,c!==a[0]))throw Error("mid");a:{c=a;e|=64;var f=c.length;if(f){var g=f-1,h=c[g];if(h!=null&&typeof h==="object"&&h.constructor===Object){b=e&128?0:-1;g-=b;if(g>=1024)throw Error("pvtlmt");for(var k in h)f=+k,f<g&&(c[f+b]=h[k],
delete h[k]);e=e&-8380417|(g&1023)<<13;break a}}if(b){k=Math.max(b,f-(e&128?0:-1));if(k>1024)throw Error("spvt");e=e&-8380417|(k&1023)<<13}}}e|=64;d===0&&(e|=2048);$d(a,e);return a}
function nf(){Kd(Sd,5)}
;function of(a,b){if(typeof a!=="object")return a;if(Array.isArray(a)){var c=a[K]|0;a.length===0&&c&1?a=void 0:c&2||(!b||4096&c||16&c?a=pf(a,c,!1,b&&!(c&16)):(ae(a,34),c&4&&Object.freeze(a)));return a}if(a[Td]===ce)return b=a.F,c=b[K]|0,de(a,c)?a:pf(b,c);if(a instanceof Fd)return a}
function pf(a,b,c,d){d!=null||(d=!!(34&b));a=gf(a,b,of,d);d=32;c&&(d|=2);b=b&8380609|d;$d(a,b);return a}
function qf(a){var b=a.F,c=b[K]|0;return de(a,c)?new a.constructor(pf(b,c,!1)):a}
function rf(a){if(a.h!==ee)return!1;var b=a.F;b=pf(b,b[K]|0);a.F=b;a.h=void 0;a.i=void 0;return!0}
function sf(a){if(!rf(a)&&de(a,a.F[K]|0))throw Error();}
;var tf=pe(0),uf={};function vf(a,b,c,d){Object.isExtensible(a);b=wf(a.F,b,c);if(b!==null||d&&a.i!==ee)return b}
function wf(a,b,c,d){if(b===-1)return null;var e=b+(c?0:-1),f=a.length-1;if(!(f<1+(c?0:-1))){if(e>=f){var g=a[f];if(g!=null&&typeof g==="object"&&g.constructor===Object){c=g[b];var h=!0}else if(e===f)c=g;else return}else c=a[e];if(d&&c!=null){d=d(c);if(d==null)return d;if(!Object.is(d,c))return h?g[b]=d:a[e]=d,d}return c}}
function xf(a,b,c,d){sf(a);var e=a.F;yf(e,e[K]|0,b,c,d);return a}
function yf(a,b,c,d,e){var f=c+(e?0:-1),g=a.length-1;if(g>=1+(e?0:-1)&&f>=g){var h=a[g];if(h!=null&&typeof h==="object"&&h.constructor===Object)return h[c]=d,b}if(f<=g)return a[f]=d,b;if(d!==void 0){var k;g=((k=b)!=null?k:b=a[K]|0)>>13&1023||536870912;c>=g?d!=null&&(f={},a[g+(e?0:-1)]=(f[c]=d,f)):a[f]=d}return b}
function zf(a){return!!(2&a)&&!!(4&a)||!!(256&a)}
function Af(a,b,c){sf(a);var d=a.F,e=d[K]|0;if(b==null)return yf(d,e,3),a;if(!Array.isArray(b))throw Jd();var f=b===Yd?7:b[K]|0,g=f,h=zf(f),k=h||Object.isFrozen(b);h||(f=0);k||(b=Ee(b),g=0,f=Bf(f,e),k=!1);f|=5;h=4&f?512&f?512:1024&f?1024:0:void 0;h=h!=null?h:0;for(var l=0;l<b.length;l++){var m=b[l],n=c(m,h);Object.is(m,n)||(k&&(b=Ee(b),g=0,f=Bf(f,e),k=!1),b[l]=n)}f!==g&&(k&&(b=Ee(b),f=Bf(f,e)),$d(b,f));yf(d,e,3,b);return a}
function Cf(a,b,c,d){sf(a);a=a.F;var e=a[K]|0;if(d==null){var f=Df(a);if(Ef(f,a,e,c)===b)f.set(c,0);else return}else{c.includes(b);f=Df(a);var g=Ef(f,a,e,c);g!==b&&(g&&(e=yf(a,e,g)),f.set(c,b))}yf(a,e,b,d)}
function Df(a){if(Ld){var b;return(b=a[Od])!=null?b:a[Od]=new Map}if(Od in a)return a[Od];b=new Map;Object.defineProperty(a,Od,{value:b});return b}
function Ef(a,b,c,d){var e=a.get(d);if(e!=null)return e;for(var f=e=0;f<d.length;f++){var g=d[f];wf(b,g)!=null&&(e!==0&&(c=yf(b,c,e)),e=g)}a.set(d,e);return e}
function Ff(a,b,c,d,e){var f=!1;a=wf(a,d,e,function(g){var h=cf(g,c,b);f=h!==g&&h!=null;return h});
if(a!=null)return f&&de(a),a}
function Gf(a,b,c,d){var e=a.F,f=e[K]|0;b=Ff(e,f,b,c,d);if(b==null)return b;f=e[K]|0;if(!de(a,f)){var g=qf(b);g!==b&&(rf(a)&&(e=a.F,f=e[K]|0),b=g,yf(e,f,c,b,d))}return b}
function Fc(a,b,c){var d=void 0===fe?2:4;var e=a.F,f=e,g=e[K]|0;e=!1;var h=de(a,g);d=h?1:d;e=!!e||d===3;var k=!h;(d===2||k)&&rf(a)&&(f=a.F,g=f[K]|0);a=wf(f,c);h=Array.isArray(a)?a:Yd;var l=h===Yd?7:h[K]|0;a=l;2&g&&(a|=2);var m=a|1;if(a=!(4&m)){var n=h,r=g,t=!!(2&m);t&&(r|=2);for(var w=!t,x=!0,z=0,G=0;z<n.length;z++){var H=cf(n[z],b,r);if(H instanceof b){if(!t){var S=de(H);w&&(w=!S);x&&(x=S)}n[G++]=H}}G<z&&(n.length=G);m|=4;m=x?m&-4097:m|4096;m=w?m|8:m&-9}m!==l&&($d(h,m),2&m&&Object.freeze(h));if(k&&
!(8&m||!h.length&&(d===1||(d!==4?0:2&m||!(16&m)&&32&g)))){zf(m)&&(h=Ee(h),m=Bf(m,g),g=yf(f,g,c,h));b=h;k=m;for(l=0;l<b.length;l++)n=b[l],m=qf(n),n!==m&&(b[l]=m);k|=8;m=k=b.length?k|4096:k&-4097;$d(h,m)}b=h;k=h=m;d===1||(d!==4?0:2&h||!(16&h)&&32&g)?zf(h)||(h|=!b.length||a&&!(4096&h)||32&g&&!(4096&h||16&h)?2:256,h!==k&&$d(b,h),Object.freeze(b)):(d===2&&zf(h)&&(b=Ee(b),k=0,h=Bf(h,g),yf(f,g,c,b)),zf(h)||(e||(h|=16),h!==k&&$d(b,h)));return b}
function Hf(a,b,c,d,e){d!=null?bf(d,b):d=void 0;xf(a,c,d,e);d&&de(d);return a}
function If(a,b,c,d){sf(a);var e=a.F,f=e[K]|0;if(d==null)return yf(e,f,c),a;if(!Array.isArray(d))throw Jd();for(var g=d===Yd?7:d[K]|0,h=g,k=zf(g),l=k||Object.isFrozen(d),m=!0,n=!0,r=0;r<d.length;r++){var t=d[r];bf(t,b);k||(t=de(t),m&&(m=!t),n&&(n=t))}k||(g=m?13:5,g=n?g&-4097:g|4096);l&&g===h||(d=Ee(d),h=0,g=Bf(g,f));g!==h&&$d(d,g);yf(e,f,c,d);return a}
function Bf(a,b){return a=(2&b?a|2:a&-3)&-273}
function Hc(a,b,c){c=c===void 0?0:c;a=Pe(vf(a,b));return a!=null?a:c}
function Jf(a,b,c){c=c===void 0?tf:c;a=vf(a,b);b=typeof a;a=a==null?a:b==="bigint"?pe(Fe(64,a)):Me(a)?b==="string"?Te(a):Ue(a):void 0;return a!=null?a:c}
function Kf(a,b,c,d){c=c===void 0?"":c;var e;return(e=af(vf(a,b,d)))!=null?e:c}
function Lf(a){var b=b===void 0?0:b;a=vf(a,1);a=a==null?a:He(a)?a|0:void 0;return a!=null?a:b}
function Mf(a,b,c){return xf(a,b,$e(c))}
function Nf(a,b,c){c=$e(c);sf(a);a=a.F;yf(a,a[K]|0,b,c===""?void 0:c,ge)}
function Of(a,b,c){if(c!=null){if(!He(c))throw Jd("enum");c|=0}return xf(a,b,c)}
;function M(a,b,c){this.F=L(a,b,c)}
M.prototype.toJSON=function(){return mf(this)};
M.prototype.serialize=function(a){return JSON.stringify(mf(this,a))};
function Pf(a,b){if(b==null||b=="")return new a;b=JSON.parse(b);if(!Array.isArray(b))throw Error("dnarr");ae(b,32);return new a(b)}
M.prototype.clone=function(){var a=this.F;return new this.constructor(pf(a,a[K]|0,!1))};
M.prototype[Td]=ce;M.prototype.toString=function(){return this.F.toString()};function Qf(){var a=Rf;this.ctor=Sf;this.isRepeated=0;this.h=Gf;this.defaultValue=void 0;this.i=a.Me!=null?ge:void 0}
Qf.prototype.register=function(){jd(this)};function Tf(a){return function(b){return Pf(a,b)}}
;function Uf(a){this.F=L(a)}
v(Uf,M);function Vf(a,b){return Af(a,b,Ne)}
;function Wf(a){this.F=L(a)}
v(Wf,M);var Xf=[1,2,3];function Yf(a){this.F=L(a)}
v(Yf,M);var Zf=[1,2,3];function $f(a){this.F=L(a)}
v($f,M);function ag(a){this.F=L(a)}
v(ag,M);function bg(a){this.F=L(a)}
v(bg,M);function cg(a){if(!a)return"";if(/^about:(?:blank|srcdoc)$/.test(a))return window.origin||"";a.indexOf("blob:")===0&&(a=a.substring(5));a=a.split("#")[0].split("?")[0];a=a.toLowerCase();a.indexOf("//")==0&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");c!=-1&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if(c!=="http"&&c!=="https"&&c!=="chrome-extension"&&
c!=="moz-extension"&&c!=="file"&&c!=="android-app"&&c!=="chrome-search"&&c!=="chrome-untrusted"&&c!=="chrome"&&c!=="app"&&c!=="devtools")throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(d!=-1){var e=b.substring(d+1);b=b.substring(0,d);if(c==="http"&&e!=="80"||c==="https"&&e!=="443")a=":"+e}return c+"://"+b+a}
;function dg(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;m=l=0}
function b(n){for(var r=g,t=0;t<64;t+=4)r[t/4]=n[t]<<24|n[t+1]<<16|n[t+2]<<8|n[t+3];for(t=16;t<80;t++)n=r[t-3]^r[t-8]^r[t-14]^r[t-16],r[t]=(n<<1|n>>>31)&4294967295;n=e[0];var w=e[1],x=e[2],z=e[3],G=e[4];for(t=0;t<80;t++){if(t<40)if(t<20){var H=z^w&(x^z);var S=1518500249}else H=w^x^z,S=1859775393;else t<60?(H=w&x|z&(w|x),S=2400959708):(H=w^x^z,S=3395469782);H=((n<<5|n>>>27)&4294967295)+H+G+S+r[t]&4294967295;G=z;z=x;x=(w<<30|w>>>2)&4294967295;w=n;n=H}e[0]=e[0]+n&4294967295;e[1]=e[1]+w&4294967295;e[2]=
e[2]+x&4294967295;e[3]=e[3]+z&4294967295;e[4]=e[4]+G&4294967295}
function c(n,r){if(typeof n==="string"){n=unescape(encodeURIComponent(n));for(var t=[],w=0,x=n.length;w<x;++w)t.push(n.charCodeAt(w));n=t}r||(r=n.length);t=0;if(l==0)for(;t+64<r;)b(n.slice(t,t+64)),t+=64,m+=64;for(;t<r;)if(f[l++]=n[t++],m++,l==64)for(l=0,b(f);t+64<r;)b(n.slice(t,t+64)),t+=64,m+=64}
function d(){var n=[],r=m*8;l<56?c(h,56-l):c(h,64-(l-56));for(var t=63;t>=56;t--)f[t]=r&255,r>>>=8;b(f);for(t=r=0;t<5;t++)for(var w=24;w>=0;w-=8)n[r++]=e[t]>>w&255;return n}
for(var e=[],f=[],g=[],h=[128],k=1;k<64;++k)h[k]=0;var l,m;a();return{reset:a,update:c,digest:d,je:function(){for(var n=d(),r="",t=0;t<n.length;t++)r+="0123456789ABCDEF".charAt(Math.floor(n[t]/16))+"0123456789ABCDEF".charAt(n[t]%16);return r}}}
;function eg(a,b,c){var d=String(D.location.href);return d&&a&&b?[b,fg(cg(d),a,c||null)].join(" "):null}
function fg(a,b,c){var d=[],e=[];if((Array.isArray(c)?2:1)==1)return e=[b,a],Tb(d,function(h){e.push(h)}),gg(e.join(" "));
var f=[],g=[];Tb(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=f.length==0?[c,b,a]:[f.join(":"),c,b,a];Tb(d,function(h){e.push(h)});
a=gg(e.join(" "));a=[c,a];g.length==0||a.push(g.join(""));return a.join("_")}
function gg(a){var b=dg();b.update(a);return b.je().toLowerCase()}
;function hg(a){this.h=a||{cookie:""}}
p=hg.prototype;p.isEnabled=function(){if(!D.navigator.cookieEnabled)return!1;if(this.h.cookie)return!0;this.set("TESTCOOKIESENABLED","1",{Xb:60});if(this.get("TESTCOOKIESENABLED")!=="1")return!1;this.remove("TESTCOOKIESENABLED");return!0};
p.set=function(a,b,c){var d=!1;if(typeof c==="object"){var e=c.cf;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.Xb}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');h===void 0&&(h=-1);c=f?";domain="+f:"";g=g?";path="+g:"";d=d?";secure":"";h=h<0?"":h==0?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+h*1E3)).toUTCString();this.h.cookie=a+"="+b+c+g+h+d+(e!=null?";samesite="+
e:"")};
p.get=function(a,b){for(var c=a+"=",d=(this.h.cookie||"").split(";"),e=0,f;e<d.length;e++){f=hb(d[e]);if(f.lastIndexOf(c,0)==0)return f.slice(c.length);if(f==a)return""}return b};
p.remove=function(a,b,c){var d=this.get(a)!==void 0;this.set(a,"",{Xb:0,path:b,domain:c});return d};
p.Tb=function(){return ig(this).keys};
p.Wa=function(){return ig(this).values};
p.clear=function(){for(var a=ig(this).keys,b=a.length-1;b>=0;b--)this.remove(a[b])};
function ig(a){a=(a.h.cookie||"").split(";");for(var b=[],c=[],d,e,f=0;f<a.length;f++)e=hb(a[f]),d=e.indexOf("="),d==-1?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));return{keys:b,values:c}}
var jg=new hg(typeof document=="undefined"?null:document);function kg(){var a=D.__SAPISID||D.__APISID||D.__3PSAPISID||D.__1PSAPISID||D.__OVERRIDE_SID;if(a)return!0;typeof document!=="undefined"&&(a=new hg(document),a=a.get("SAPISID")||a.get("APISID")||a.get("__Secure-3PAPISID")||a.get("__Secure-1PAPISID"));return!!a}
function lg(a,b,c,d){(a=D[a])||typeof document==="undefined"||(a=(new hg(document)).get(b));return a?eg(a,c,d):null}
function mg(a){var b=cg(String(D.location.href)),c=[];if(kg()){b=b.indexOf("https:")==0||b.indexOf("chrome-extension:")==0||b.indexOf("chrome-untrusted://new-tab-page")==0||b.indexOf("moz-extension:")==0;var d=b?D.__SAPISID:D.__APISID;d||typeof document==="undefined"||(d=new hg(document),d=d.get(b?"SAPISID":"APISID")||d.get("__Secure-3PAPISID"));(d=d?eg(d,b?"SAPISIDHASH":"APISIDHASH",a):null)&&c.push(d);b&&((b=lg("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&c.push(b),(a=lg("__3PSAPISID",
"__Secure-3PAPISID","SAPISID3PHASH",a))&&c.push(a))}return c.length==0?null:c.join(" ")}
;function ng(){}
ng.prototype.compress=function(a){var b,c,d,e;return B(function(f){switch(f.h){case 1:return b=new CompressionStream("gzip"),c=(new Response(b.readable)).arrayBuffer(),d=b.writable.getWriter(),f.yield(d.write((new TextEncoder).encode(a)),2);case 2:return f.yield(d.close(),3);case 3:return e=Uint8Array,f.yield(c,4);case 4:return f.return(new e(f.i))}})};
ng.prototype.isSupported=function(a){return a<1024?!1:typeof CompressionStream!=="undefined"};function og(a){this.F=L(a)}
v(og,M);function pg(a,b){this.intervalMs=a;this.callback=b;this.enabled=!1;this.h=function(){return Ya()};
this.i=this.h()}
pg.prototype.setInterval=function(a){this.intervalMs=a;this.timer&&this.enabled?(this.stop(),this.start()):this.timer&&this.stop()};
pg.prototype.start=function(){var a=this;this.enabled=!0;this.timer||(this.timer=setTimeout(function(){a.tick()},this.intervalMs),this.i=this.h())};
pg.prototype.stop=function(){this.enabled=!1;this.timer&&(clearTimeout(this.timer),this.timer=void 0)};
pg.prototype.tick=function(){var a=this;if(this.enabled){var b=Math.max(this.h()-this.i,0);b<this.intervalMs*.8?this.timer=setTimeout(function(){a.tick()},this.intervalMs-b):(this.timer&&(clearTimeout(this.timer),this.timer=void 0),this.callback(),this.enabled&&(this.stop(),this.start()))}else this.timer=void 0};function qg(a){this.F=L(a)}
v(qg,M);function rg(a){this.F=L(a)}
v(rg,M);function sg(a,b){this.x=a!==void 0?a:0;this.y=b!==void 0?b:0}
p=sg.prototype;p.clone=function(){return new sg(this.x,this.y)};
p.equals=function(a){return a instanceof sg&&(this==a?!0:this&&a?this.x==a.x&&this.y==a.y:!1)};
p.ceil=function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this};
p.floor=function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this};
p.round=function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this};
p.scale=function(a,b){this.x*=a;this.y*=typeof b==="number"?b:a;return this};function tg(a,b){this.width=a;this.height=b}
p=tg.prototype;p.clone=function(){return new tg(this.width,this.height)};
p.aspectRatio=function(){return this.width/this.height};
p.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
p.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
p.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};
p.scale=function(a,b){this.width*=a;this.height*=typeof b==="number"?b:a;return this};function ug(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function vg(a){var b=[],c=0,d;for(d in a)b[c++]=a[d];return b}
function wg(a){var b=xg,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function yg(a){for(var b in a)return!1;return!0}
function zg(a,b){if(a!==null&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function Ag(a){return a!==null&&"privembed"in a?a.privembed:!1}
function Bg(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function Cg(a){var b={},c;for(c in a)b[c]=a[c];return b}
function Dg(a){if(!a||typeof a!=="object")return a;if(typeof a.clone==="function")return a.clone();if(typeof Map!=="undefined"&&a instanceof Map)return new Map(a);if(typeof Set!=="undefined"&&a instanceof Set)return new Set(a);if(a instanceof Date)return new Date(a.getTime());var b=Array.isArray(a)?[]:typeof ArrayBuffer!=="function"||typeof ArrayBuffer.isView!=="function"||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=Dg(a[c]);return b}
var Eg="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function Fg(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<Eg.length;f++)c=Eg[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;function Gg(a,b){this.h=a===Hg&&b||""}
Gg.prototype.toString=function(){return this.h};
var Hg={};new Gg(Hg,"");"ARTICLE SECTION NAV ASIDE H1 H2 H3 H4 H5 H6 HEADER FOOTER ADDRESS P HR PRE BLOCKQUOTE OL UL LH LI DL DT DD FIGURE FIGCAPTION MAIN DIV EM STRONG SMALL S CITE Q DFN ABBR RUBY RB RT RTC RP DATA TIME CODE VAR SAMP KBD SUB SUP I B U MARK BDI BDO SPAN BR WBR NOBR INS DEL PICTURE PARAM TRACK MAP TABLE CAPTION COLGROUP COL TBODY THEAD TFOOT TR TD TH SELECT DATALIST OPTGROUP OPTION OUTPUT PROGRESS METER FIELDSET LEGEND DETAILS SUMMARY MENU DIALOG SLOT CANVAS FONT CENTER ACRONYM BASEFONT BIG DIR HGROUP STRIKE TT".split(" ").concat(["BUTTON",
"INPUT"]);function Ig(a){var b=document;return typeof a==="string"?b.getElementById(a):a}
function Jg(a){var b=document;a=String(a);b.contentType==="application/xhtml+xml"&&(a=a.toLowerCase());return b.createElement(a)}
function Kg(a){a&&a.parentNode&&a.parentNode.removeChild(a)}
function Lg(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function Mg(a){this.F=L(a)}
v(Mg,M);Mg.prototype.qc=function(){return Lf(this)};function Ng(a){this.F=L(a)}
v(Ng,M);function Og(a){this.F=L(a)}
v(Og,M);function Pg(a){If(Qg,Ng,1,a)}
var Rg=Tf(Og);function Sg(a){this.F=L(a)}
v(Sg,M);var Tg=["platform","platformVersion","architecture","model","uaFullVersion"],Qg=new Og,Ug=null;function Vg(a,b){b=b===void 0?Tg:b;if(!Ug){var c;a=(c=a.navigator)==null?void 0:c.userAgentData;if(!a||typeof a.getHighEntropyValues!=="function"||a.brands&&typeof a.brands.map!=="function")return Promise.reject(Error("UACH unavailable"));Pg((a.brands||[]).map(function(e){var f=new Ng;f=Mf(f,1,e.brand);return Mf(f,2,e.version)}));
typeof a.mobile==="boolean"&&xf(Qg,2,Ke(a.mobile));Ug=a.getHighEntropyValues(b)}var d=new Set(b);return Ug.then(function(e){var f=Qg.clone();d.has("platform")&&Mf(f,3,e.platform);d.has("platformVersion")&&Mf(f,4,e.platformVersion);d.has("architecture")&&Mf(f,5,e.architecture);d.has("model")&&Mf(f,6,e.model);d.has("uaFullVersion")&&Mf(f,7,e.uaFullVersion);return f.serialize()}).catch(function(){return Qg.serialize()})}
;function Wg(a){this.F=L(a)}
v(Wg,M);function Xg(a){return Of(a,1,1)}
;function Yg(a){this.F=L(a,4)}
v(Yg,M);function Zg(a){this.F=L(a,36)}
v(Zg,M);function $g(a){this.F=L(a,19)}
v($g,M);$g.prototype.ac=function(a){return Of(this,2,a)};function ah(a,b){this.Oa=b=b===void 0?!1:b;this.j=this.locale=null;this.i=0;this.isFinal=!1;this.h=new $g;Number.isInteger(a)&&this.h.ac(a);b||(this.locale=document.documentElement.getAttribute("lang"));bh(this,new Wg)}
ah.prototype.ac=function(a){this.h.ac(a);return this};
function bh(a,b){Hf(a.h,Wg,1,b);Lf(b)||Xg(b);a.Oa||(b=ch(a),Kf(b,5)||Mf(b,5,a.locale));a.j&&(b=ch(a),Gf(b,Og,9)||Hf(b,Og,9,a.j))}
function dh(a,b){a.i=b}
function eh(a){var b=b===void 0?Tg:b;var c=a.Oa?void 0:window;c?Vg(c,b).then(function(d){a.j=Rg(d!=null?d:"[]");d=ch(a);Hf(d,Og,9,a.j);return!0}).catch(function(){return!1}):Promise.resolve(!1)}
function ch(a){a=Gf(a.h,Wg,1);var b=Gf(a,Sg,11);b||(b=new Sg,Hf(a,Sg,11,b));return b}
function fh(a,b,c,d,e,f,g){c=c===void 0?0:c;d=d===void 0?0:d;e=e===void 0?null:e;f=f===void 0?0:f;g=g===void 0?0:g;if(!a.Oa){var h=ch(a);var k=new Mg;k=Of(k,1,a.i);k=xf(k,2,Ke(a.isFinal));d=xf(k,3,Oe(d>0?d:void 0));d=xf(d,4,Oe(f>0?f:void 0));d=xf(d,5,Oe(g>0?g:void 0));f=d.F;g=f[K]|0;d=de(d,g)?d:new d.constructor(pf(f,g,!0));Hf(h,Mg,10,d)}a=a.h.clone();h=Date.now().toString();a=xf(a,4,We(h));b=b.slice();b=If(a,Zg,3,b);e&&(a=new qg,e=xf(a,13,Oe(e)),a=new rg,e=Hf(a,qg,2,e),a=new Yg,e=Hf(a,rg,1,e),e=
Of(e,2,9),Hf(b,Yg,18,e));c&&xf(b,14,We(c));return b}
;var gh=function(){if(!D.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{var c=function(){};
D.addEventListener("test",c,b);D.removeEventListener("test",c,b)}catch(d){}return a}();function hh(a){this.h=this.i=this.j=a}
hh.prototype.reset=function(){this.h=this.i=this.j};
hh.prototype.getValue=function(){return this.i};function Rf(a){this.F=L(a,8)}
v(Rf,M);var ih=Tf(Rf);function Sf(a){this.F=L(a)}
v(Sf,M);var jh;jh=new Qf;function kh(a){I.call(this);var b=this;this.componentId="";this.h=[];this.Ra="";this.pageId=null;this.fb=this.ma=-1;this.G=this.experimentIds=null;this.A=this.o=0;this.U=null;this.Z=this.ha=0;this.Lb=1;this.timeoutMillis=0;this.xa=!1;this.logSource=a.logSource;this.yb=a.yb||function(){};
this.j=new ah(a.logSource,a.Oa);this.network=a.network||null;this.ob=a.ob||null;this.bufferSize=1E3;this.P=a.Af||null;this.sessionIndex=a.sessionIndex||null;this.Rb=a.Rb||!1;this.logger=null;this.withCredentials=!a.sd;this.Oa=a.Oa||!1;this.Y=!this.Oa&&!!window&&!!window.navigator&&window.navigator.sendBeacon!==void 0;this.Qa=typeof URLSearchParams!=="undefined"&&!!(new URL(lh())).searchParams&&!!(new URL(lh())).searchParams.set;var c=Xg(new Wg);bh(this.j,c);this.u=new hh(1E4);a=mh(this,a.od);this.i=
new pg(this.u.getValue(),a);this.Fa=new pg(6E5,a);this.Rb||this.Fa.start();this.Oa||(document.addEventListener("visibilitychange",function(){if(document.visibilityState==="hidden"){nh(b);var d;(d=b.U)==null||d.flush()}}),document.addEventListener("pagehide",function(){nh(b);
var d;(d=b.U)==null||d.flush()}))}
v(kh,I);function mh(a,b){return a.Qa?b?function(){b().then(function(){a.flush()})}:function(){a.flush()}:function(){}}
kh.prototype.ba=function(){nh(this);this.i.stop();this.Fa.stop();I.prototype.ba.call(this)};
function oh(a){a.P||(a.P=lh());try{return(new URL(a.P)).toString()}catch(b){return(new URL(a.P,window.location.origin)).toString()}}
function ph(a,b,c){a.U&&a.U.kb(b,c)}
kh.prototype.log=function(a){ph(this,2,1);if(this.Qa){a=a.clone();var b=this.Lb++;a=xf(a,21,We(b));this.componentId&&Mf(a,26,this.componentId);b=a;var c=vf(b,1);var d=d===void 0?!1:d;var e=typeof c;d=c==null?c:e==="bigint"?String(Fe(64,c)):Me(c)?e==="string"?Re(c):d?Se(c):Ve(c):void 0;d==null&&(d=Date.now(),d=Number.isFinite(d)?d.toString():"0",xf(b,1,We(d)));d=vf(b,15);d!=null&&(typeof d==="bigint"?ve(d)?d=Number(d):(d=Fe(64,d),d=ve(d)?Number(d):String(d)):d=Me(d)?typeof d==="number"?Ve(d):Re(d):
void 0);d==null&&xf(b,15,We((new Date).getTimezoneOffset()*60));this.experimentIds&&(d=this.experimentIds.clone(),Hf(b,og,16,d));ph(this,1,1);b=this.h.length-this.bufferSize+1;b>0&&(this.h.splice(0,b),this.o+=b,ph(this,3,b));this.h.push(a);this.Rb||this.i.enabled||this.i.start()}};
kh.prototype.flush=function(a,b){var c=this;if(this.h.length===0)a&&a();else if(this.xa&&this.Y)this.j.i=3,qh(this);else{var d=Date.now();if(this.fb>d&&this.ma<d)b&&b("throttled");else{this.network&&(typeof this.network.qc==="function"?dh(this.j,this.network.qc()):this.j.i=0);var e=this.h.length,f=fh(this.j,this.h,this.o,this.A,this.ob,this.ha,this.Z),g=this.yb();if(g&&this.Ra===g)b&&b("stale-auth-token");else{this.h=[];this.i.enabled&&this.i.stop();this.o=0;d=f.serialize();var h;this.G&&this.G.isSupported(d.length)&&
(h=this.G.compress(d));var k=rh(this,d,g),l=function(r){c.u.reset();c.i.setInterval(c.u.getValue());if(r){var t=null;try{var w=JSON.stringify(JSON.parse(r.replace(")]}'\n","")));t=ih(w)}catch(G){}if(t){r=Number(Jf(t,1,pe("-1")));r>0&&(c.ma=Date.now(),c.fb=c.ma+r);r=Za(Pd);var x;Ld&&r&&((x=t.F[r])==null?void 0:x[175237375])!=null&&Kd(Qd,3);a:{var z=z===void 0?!1:z;if(Za(Vd)&&Za(Pd)&&void 0===Vd){x=t.F;r=x[Pd];if(!r)break a;if(r=r.fi)try{r(x,175237375,ff);break a}catch(G){Vc(G)}}z&&(z=t.F,(x=Za(Pd))&&
x in z&&(z=z[x])&&delete z[175237375])}z=jh.ctor?jh.h(t,jh.ctor,175237375,jh.i):jh.h(t,175237375,null,jh.i);if(z=z===null?void 0:z)z=Hc(z,1,-1),z!==-1&&(c.u=new hh(z<1?1:z),c.i.setInterval(c.u.getValue()))}}a&&a();c.A=0},m=function(r,t){var w=Fc(f,Zg,3);
var x=Number(Jf(f,14)),z=c.u;z.h=Math.min(3E5,z.h*2);z.i=Math.min(3E5,z.h+Math.round(.1*(Math.random()-.5)*2*z.h));c.i.setInterval(c.u.getValue());r===401&&g&&(c.Ra=g);x&&(c.o+=x);t===void 0&&(t=c.isRetryable(r));t&&(c.h=w.concat(c.h),c.Rb||c.i.enabled||c.i.start());ph(c,7,1);b&&b("net-send-failed",r);++c.A},n=function(){c.network&&c.network.send(k,l,m)};
h?h.then(function(r){ph(c,5,e);k.Ec["Content-Encoding"]="gzip";k.Ec["Content-Type"]="application/binary";k.body=r;k.ce=2;n()},function(){ph(c,6,e);
n()}):n()}}}};
function rh(a,b,c){c=c===void 0?null:c;var d=d===void 0?a.withCredentials:d;var e={},f=new URL(oh(a));c&&(e.Authorization=c);a.sessionIndex&&(e["X-Goog-AuthUser"]=a.sessionIndex,f.searchParams.set("authuser",a.sessionIndex));a.pageId&&(Object.defineProperty(e,"X-Goog-PageId",{value:a.pageId}),f.searchParams.set("pageId",a.pageId));return{url:f.toString(),body:b,ce:1,Ec:e,requestType:"POST",withCredentials:d,timeoutMillis:a.timeoutMillis}}
function nh(a){a.j.isFinal=!0;a.flush();a.j.isFinal=!1}
function qh(a){sh(a,function(b,c){b=new URL(b);b.searchParams.set("format","json");var d=!1;try{d=window.navigator.sendBeacon(b.toString(),c.serialize())}catch(e){}d||(a.Y=!1);return d})}
function sh(a,b){if(a.h.length!==0){var c=new URL(oh(a));c.searchParams.delete("format");var d=a.yb();d&&c.searchParams.set("auth",d);c.searchParams.set("authuser",a.sessionIndex||"0");for(d=0;d<10&&a.h.length;++d){var e=a.h.slice(0,32),f=fh(a.j,e,a.o,a.A,a.ob,a.ha,a.Z);if(!b(c.toString(),f)){++a.A;break}a.o=0;a.A=0;a.ha=0;a.Z=0;a.h=a.h.slice(e.length)}a.i.enabled&&a.i.stop()}}
kh.prototype.isRetryable=function(a){return 500<=a&&a<600||a===401||a===0};
function lh(){return"https://play.google.com/log?format=json&hasfast=true"}
;function uh(){this.Wd=typeof AbortController!=="undefined"}
uh.prototype.send=function(a,b,c){var d=this,e,f,g,h,k,l,m,n,r,t;return B(function(w){switch(w.h){case 1:return f=(e=d.Wd?new AbortController:void 0)?setTimeout(function(){e.abort()},a.timeoutMillis):void 0,va(w,2,3),g=Object.assign({},{method:a.requestType,
headers:Object.assign({},a.Ec)},a.body&&{body:a.body},a.withCredentials&&{credentials:"include"},{signal:a.timeoutMillis&&e?e.signal:null}),w.yield(fetch(a.url,g),5);case 5:h=w.i;if(h.status!==200){(k=c)==null||k(h.status);w.B(3);break}if((l=b)==null){w.B(7);break}return w.yield(h.text(),8);case 8:l(w.i);case 7:case 3:w.P=[w.j];w.H=0;w.o=0;clearTimeout(f);ya(w);break;case 2:m=xa(w);switch((n=m)==null?void 0:n.name){case "AbortError":(r=c)==null||r(408);break;default:(t=c)==null||t(400)}w.B(3)}})};
uh.prototype.qc=function(){return 4};function vh(a,b){b=b===void 0?"0":b;I.call(this);this.logSource=a;this.sessionIndex=b;this.Va="https://play.google.com/log?format=json&hasfast=true";this.i=null;this.o=!1;this.network=null;this.componentId="";this.h=this.ob=null;this.j=!1;this.pageId=null;this.bufferSize=void 0}
v(vh,I);function wh(a,b){a.i=b;return a}
function xh(a,b){a.network=b;return a}
function yh(a,b){a.h=b}
function zh(a){a.j=!0;return a}
vh.prototype.sd=function(){this.u=!0;return this};
function Ah(a){a.network||(a.network=new uh);var b=new kh({logSource:a.logSource,yb:a.yb?a.yb:mg,sessionIndex:a.sessionIndex,Af:a.Va,Oa:a.o,Rb:!1,sd:a.u,od:a.od,network:a.network});zc(a,b);if(a.i){var c=a.i,d=ch(b.j);Mf(d,7,c)}b.G=new ng;a.componentId&&(b.componentId=a.componentId);a.ob&&(b.ob=a.ob);a.pageId&&(b.pageId=a.pageId);a.h&&((d=a.h)?(b.experimentIds||(b.experimentIds=new og),c=b.experimentIds,d=d.serialize(),Mf(c,4,d)):b.experimentIds&&xf(b.experimentIds,4));a.j&&(b.xa=b.Y);eh(b.j);a.bufferSize&&
(b.bufferSize=a.bufferSize);a.network.ac&&a.network.ac(a.logSource);a.network.pf&&a.network.pf(b);return b}
;function Bh(a,b,c,d,e,f,g){a=a===void 0?-1:a;b=b===void 0?"":b;c=c===void 0?"":c;d=d===void 0?!1:d;e=e===void 0?"":e;I.call(this);this.logSource=a;this.componentId=b;f?b=f:(a=new vh(a,"0"),a.componentId=b,zc(this,a),c!==""&&(a.Va=c),d&&(a.o=!0),e&&wh(a,e),g&&xh(a,g),b=Ah(a));this.h=b}
v(Bh,I);
Bh.prototype.flush=function(a){var b=a||[];if(b.length){a=new bg;for(var c=[],d=0;d<b.length;d++){var e=b[d],f=new ag;f=Mf(f,1,e.i);var g=Ch(e);f=Af(f,g,Ze);g=[];var h=[];for(var k=y(e.h.keys()),l=k.next();!l.done;l=k.next())h.push(l.value.split(","));for(k=0;k<h.length;k++){l=h[k];var m=e.o;for(var n=e.Nc(l)||[],r=[],t=0;t<n.length;t++){var w=n[t],x=w&&w.h;w=new Yf;switch(m){case 3:x=Number(x);Number.isFinite(x)&&Cf(w,1,Zf,We(x));break;case 2:x=Number(x);if(x!=null&&typeof x!=="number")throw Error("Value of float/double field must be a number, found "+typeof x+
": "+x);Cf(w,2,Zf,x)}r.push(w)}m=r;for(n=0;n<m.length;n++){r=m[n];t=new $f;r=Hf(t,Yf,2,r);t=l;w=[];x=Dh(e);for(var z=0;z<x.length;z++){var G=x[z],H=t[z],S=new Wf;switch(G){case 3:Cf(S,1,Xf,$e(String(H)));break;case 2:G=Number(H);Number.isFinite(G)&&Cf(S,2,Xf,Oe(G));break;case 1:Cf(S,3,Xf,Ke(H==="true"))}w.push(S)}If(r,Wf,1,w);g.push(r)}}If(f,$f,4,g);c.push(f);e.clear()}If(a,ag,1,c);b=this.h;if(a instanceof Zg)b.log(a);else try{var Z=new Zg,mb=a.serialize();var Rb=Mf(Z,8,mb);b.log(Rb)}catch(Wa){ph(b,
4,1)}this.h.flush()}};function Eh(a){this.h=a}
;function Fh(a,b,c){this.i=a;this.o=b;this.fields=c||[];this.h=new Map}
function Dh(a){return a.fields.map(function(b){return b.fieldType})}
function Ch(a){return a.fields.map(function(b){return b.fieldName})}
p=Fh.prototype;p.Xd=function(a){var b=C.apply(1,arguments),c=this.Nc(b);c?c.push(new Eh(a)):this.Kd(a,b)};
p.Kd=function(a){var b=this.nd(C.apply(1,arguments));this.h.set(b,[new Eh(a)])};
p.Nc=function(){var a=this.nd(C.apply(0,arguments));return this.h.has(a)?this.h.get(a):void 0};
p.we=function(){var a=this.Nc(C.apply(0,arguments));return a&&a.length?a[0]:void 0};
p.clear=function(){this.h.clear()};
p.nd=function(){var a=C.apply(0,arguments);return a?a.join(","):"key"};function Gh(a,b){Fh.call(this,a,3,b)}
v(Gh,Fh);Gh.prototype.j=function(a){var b=C.apply(1,arguments),c=0,d=this.we(b);d&&(c=d.h);this.Kd(c+a,b)};function Hh(a,b){Fh.call(this,a,2,b)}
v(Hh,Fh);Hh.prototype.record=function(a){this.Xd(a,C.apply(1,arguments))};function Ih(a,b){this.type=a;this.h=this.target=b;this.defaultPrevented=this.j=!1}
Ih.prototype.stopPropagation=function(){this.j=!0};
Ih.prototype.preventDefault=function(){this.defaultPrevented=!0};function Jh(a,b){Ih.call(this,a?a.type:"");this.relatedTarget=this.h=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.i=null;a&&this.init(a,b)}
$a(Jh,Ih);
Jh.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.h=b;b=a.relatedTarget;b||(c=="mouseover"?b=a.fromElement:c=="mouseout"&&(b=a.toElement));this.relatedTarget=b;d?(this.clientX=d.clientX!==void 0?d.clientX:d.pageX,this.clientY=d.clientY!==void 0?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||0):(this.clientX=a.clientX!==void 0?a.clientX:a.pageX,this.clientY=a.clientY!==
void 0?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||(c=="keypress"?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType=a.pointerType;this.state=a.state;this.i=a;a.defaultPrevented&&Jh.Aa.preventDefault.call(this)};
Jh.prototype.stopPropagation=function(){Jh.Aa.stopPropagation.call(this);this.i.stopPropagation?this.i.stopPropagation():this.i.cancelBubble=!0};
Jh.prototype.preventDefault=function(){Jh.Aa.preventDefault.call(this);var a=this.i;a.preventDefault?a.preventDefault():a.returnValue=!1};var Kh="closure_listenable_"+(Math.random()*1E6|0);var Lh=0;function Mh(a,b,c,d,e){this.listener=a;this.proxy=null;this.src=b;this.type=c;this.capture=!!d;this.uc=e;this.key=++Lh;this.Zb=this.jc=!1}
function Nh(a){a.Zb=!0;a.listener=null;a.proxy=null;a.src=null;a.uc=null}
;function Oh(a){this.src=a;this.listeners={};this.h=0}
Oh.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.h++);var g=Ph(a,b,d,e);g>-1?(b=a[g],c||(b.jc=!1)):(b=new Mh(b,this.src,f,!!d,e),b.jc=c,a.push(b));return b};
Oh.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=Ph(e,b,c,d);return b>-1?(Nh(e[b]),Array.prototype.splice.call(e,b,1),e.length==0&&(delete this.listeners[a],this.h--),!0):!1};
function Qh(a,b){var c=b.type;c in a.listeners&&Yb(a.listeners[c],b)&&(Nh(b),a.listeners[c].length==0&&(delete a.listeners[c],a.h--))}
function Ph(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.Zb&&f.listener==b&&f.capture==!!c&&f.uc==d)return e}return-1}
;var Rh="closure_lm_"+(Math.random()*1E6|0),Sh={},Th=0;function Uh(a,b,c,d,e){if(d&&d.once)Vh(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)Uh(a,b[f],c,d,e);else c=Wh(c),a&&a[Kh]?a.listen(b,c,Oa(d)?!!d.capture:!!d,e):Xh(a,b,c,!1,d,e)}
function Xh(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=Oa(e)?!!e.capture:!!e,h=Yh(a);h||(a[Rh]=h=new Oh(a));c=h.add(b,c,d,g,f);if(!c.proxy){d=Zh();c.proxy=d;d.src=a;d.listener=c;if(a.addEventListener)gh||(e=g),e===void 0&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent($h(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");Th++}}
function Zh(){function a(c){return b.call(a.src,a.listener,c)}
var b=ai;return a}
function Vh(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)Vh(a,b[f],c,d,e);else c=Wh(c),a&&a[Kh]?bi(a,b,c,Oa(d)?!!d.capture:!!d,e):Xh(a,b,c,!0,d,e)}
function ci(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)ci(a,b[f],c,d,e);else(d=Oa(d)?!!d.capture:!!d,c=Wh(c),a&&a[Kh])?a.i.remove(String(b),c,d,e):a&&(a=Yh(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=Ph(b,c,d,e)),(c=a>-1?b[a]:null)&&di(c))}
function di(a){if(typeof a!=="number"&&a&&!a.Zb){var b=a.src;if(b&&b[Kh])Qh(b.i,a);else{var c=a.type,d=a.proxy;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent($h(c),d):b.addListener&&b.removeListener&&b.removeListener(d);Th--;(c=Yh(b))?(Qh(c,a),c.h==0&&(c.src=null,b[Rh]=null)):Nh(a)}}}
function $h(a){return a in Sh?Sh[a]:Sh[a]="on"+a}
function ai(a,b){if(a.Zb)a=!0;else{b=new Jh(b,this);var c=a.listener,d=a.uc||a.src;a.jc&&di(a);a=c.call(d,b)}return a}
function Yh(a){a=a[Rh];return a instanceof Oh?a:null}
var ei="__closure_events_fn_"+(Math.random()*1E9>>>0);function Wh(a){if(typeof a==="function")return a;a[ei]||(a[ei]=function(b){return a.handleEvent(b)});
return a[ei]}
;function fi(){I.call(this);this.i=new Oh(this);this.xa=this;this.Z=null}
$a(fi,I);fi.prototype[Kh]=!0;p=fi.prototype;p.addEventListener=function(a,b,c,d){Uh(this,a,b,c,d)};
p.removeEventListener=function(a,b,c,d){ci(this,a,b,c,d)};
function gi(a,b){var c=a.Z;if(c){var d=[];for(var e=1;c;c=c.Z)d.push(c),++e}a=a.xa;c=b.type||b;typeof b==="string"?b=new Ih(b,a):b instanceof Ih?b.target=b.target||a:(e=b,b=new Ih(c,a),Fg(b,e));e=!0;var f;if(d)for(f=d.length-1;!b.j&&f>=0;f--){var g=b.h=d[f];e=hi(g,c,!0,b)&&e}b.j||(g=b.h=a,e=hi(g,c,!0,b)&&e,b.j||(e=hi(g,c,!1,b)&&e));if(d)for(f=0;!b.j&&f<d.length;f++)g=b.h=d[f],e=hi(g,c,!1,b)&&e}
p.ba=function(){fi.Aa.ba.call(this);this.removeAllListeners();this.Z=null};
p.listen=function(a,b,c,d){return this.i.add(String(a),b,!1,c,d)};
function bi(a,b,c,d,e){a.i.add(String(b),c,!0,d,e)}
p.removeAllListeners=function(a){if(this.i){var b=this.i;a=a&&a.toString();var c=0,d;for(d in b.listeners)if(!a||d==a){for(var e=b.listeners[d],f=0;f<e.length;f++)++c,Nh(e[f]);delete b.listeners[d];b.h--}b=c}else b=0;return b};
function hi(a,b,c,d){b=a.i.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.Zb&&g.capture==c){var h=g.listener,k=g.uc||g.src;g.jc&&Qh(a.i,g);e=h.call(k,d)!==!1&&e}}return e&&!d.defaultPrevented}
;var ii=typeof AsyncContext!=="undefined"&&typeof AsyncContext.Snapshot==="function"?function(a){return a&&AsyncContext.Snapshot.wrap(a)}:function(a){return a};function ji(a,b){this.j=a;this.o=b;this.i=0;this.h=null}
ji.prototype.get=function(){if(this.i>0){this.i--;var a=this.h;this.h=a.next;a.next=null}else a=this.j();return a};
function ki(a,b){a.o(b);a.i<100&&(a.i++,b.next=a.h,a.h=b)}
;function li(){this.i=this.h=null}
li.prototype.add=function(a,b){var c=mi.get();c.set(a,b);this.i?this.i.next=c:this.h=c;this.i=c};
li.prototype.remove=function(){var a=null;this.h&&(a=this.h,this.h=this.h.next,this.h||(this.i=null),a.next=null);return a};
var mi=new ji(function(){return new ni},function(a){return a.reset()});
function ni(){this.next=this.scope=this.h=null}
ni.prototype.set=function(a,b){this.h=a;this.scope=b;this.next=null};
ni.prototype.reset=function(){this.next=this.scope=this.h=null};var oi,pi=!1,qi=new li;function ri(a,b){oi||si();pi||(oi(),pi=!0);qi.add(a,b)}
function si(){var a=Promise.resolve(void 0);oi=function(){a.then(ti)}}
function ti(){for(var a;a=qi.remove();){try{a.h.call(a.scope)}catch(b){Vc(b)}ki(mi,a)}pi=!1}
;function ui(){}
function vi(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;function wi(a){this.X=0;this.bb=void 0;this.vb=this.Ta=this.parent_=null;this.sc=this.Mc=!1;if(a!=ui)try{var b=this;a.call(void 0,function(c){xi(b,2,c)},function(c){xi(b,3,c)})}catch(c){xi(this,3,c)}}
function yi(){this.next=this.context=this.h=this.i=this.child=null;this.j=!1}
yi.prototype.reset=function(){this.context=this.h=this.i=this.child=null;this.j=!1};
var zi=new ji(function(){return new yi},function(a){a.reset()});
function Ai(a,b,c){var d=zi.get();d.i=a;d.h=b;d.context=c;return d}
function Bi(a){return new wi(function(b,c){c(a)})}
wi.prototype.then=function(a,b,c){return Ci(this,ii(typeof a==="function"?a:null),ii(typeof b==="function"?b:null),c)};
wi.prototype.$goog_Thenable=!0;function Di(a,b,c,d){Ei(a,Ai(b||ui,c||null,d))}
p=wi.prototype;p.finally=function(a){var b=this;a=ii(a);return new Promise(function(c,d){Di(b,function(e){a();c(e)},function(e){a();
d(e)})})};
p.Gc=function(a,b){return Ci(this,null,ii(a),b)};
p.catch=wi.prototype.Gc;p.cancel=function(a){if(this.X==0){var b=new Fi(a);ri(function(){Gi(this,b)},this)}};
function Gi(a,b){if(a.X==0)if(a.parent_){var c=a.parent_;if(c.Ta){for(var d=0,e=null,f=null,g=c.Ta;g&&(g.j||(d++,g.child==a&&(e=g),!(e&&d>1)));g=g.next)e||(f=g);e&&(c.X==0&&d==1?Gi(c,b):(f?(d=f,d.next==c.vb&&(c.vb=d),d.next=d.next.next):Hi(c),Ii(c,e,3,b)))}a.parent_=null}else xi(a,3,b)}
function Ei(a,b){a.Ta||a.X!=2&&a.X!=3||Ji(a);a.vb?a.vb.next=b:a.Ta=b;a.vb=b}
function Ci(a,b,c,d){var e=Ai(null,null,null);e.child=new wi(function(f,g){e.i=b?function(h){try{var k=b.call(d,h);f(k)}catch(l){g(l)}}:f;
e.h=c?function(h){try{var k=c.call(d,h);k===void 0&&h instanceof Fi?g(h):f(k)}catch(l){g(l)}}:g});
e.child.parent_=a;Ei(a,e);return e.child}
p.yf=function(a){this.X=0;xi(this,2,a)};
p.zf=function(a){this.X=0;xi(this,3,a)};
function xi(a,b,c){if(a.X==0){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.X=1;a:{var d=c,e=a.yf,f=a.zf;if(d instanceof wi){Di(d,e,f,a);var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(l){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(Oa(d))try{var k=d.then;if(typeof k==="function"){Ki(d,k,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}}g||(a.bb=c,a.X=b,a.parent_=null,Ji(a),b!=3||c instanceof Fi||Li(a,c))}}
function Ki(a,b,c,d,e){function f(k){h||(h=!0,d.call(e,k))}
function g(k){h||(h=!0,c.call(e,k))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function Ji(a){a.Mc||(a.Mc=!0,ri(a.qe,a))}
function Hi(a){var b=null;a.Ta&&(b=a.Ta,a.Ta=b.next,b.next=null);a.Ta||(a.vb=null);return b}
p.qe=function(){for(var a;a=Hi(this);)Ii(this,a,this.X,this.bb);this.Mc=!1};
function Ii(a,b,c,d){if(c==3&&b.h&&!b.j)for(;a&&a.sc;a=a.parent_)a.sc=!1;if(b.child)b.child.parent_=null,Mi(b,c,d);else try{b.j?b.i.call(b.context):Mi(b,c,d)}catch(e){Ni.call(null,e)}ki(zi,b)}
function Mi(a,b,c){b==2?a.i.call(a.context,c):a.h&&a.h.call(a.context,c)}
function Li(a,b){a.sc=!0;ri(function(){a.sc&&Ni.call(null,b)})}
var Ni=Vc;function Fi(a){fb.call(this,a)}
$a(Fi,fb);Fi.prototype.name="cancel";function Oi(a,b){fi.call(this);this.j=a||1;this.h=b||D;this.o=Ua(this.uf,this);this.u=Ya()}
$a(Oi,fi);p=Oi.prototype;p.enabled=!1;p.Ea=null;p.setInterval=function(a){this.j=a;this.Ea&&this.enabled?(this.stop(),this.start()):this.Ea&&this.stop()};
p.uf=function(){if(this.enabled){var a=Ya()-this.u;a>0&&a<this.j*.8?this.Ea=this.h.setTimeout(this.o,this.j-a):(this.Ea&&(this.h.clearTimeout(this.Ea),this.Ea=null),gi(this,"tick"),this.enabled&&(this.stop(),this.start()))}};
p.start=function(){this.enabled=!0;this.Ea||(this.Ea=this.h.setTimeout(this.o,this.j),this.u=Ya())};
p.stop=function(){this.enabled=!1;this.Ea&&(this.h.clearTimeout(this.Ea),this.Ea=null)};
p.ba=function(){Oi.Aa.ba.call(this);this.stop();delete this.h};function Pi(a){I.call(this);this.G=a;this.j=0;this.o=100;this.u=!1;this.i=new Map;this.A=new Set;this.flushInterval=3E4;this.h=new Oi(this.flushInterval);this.h.listen("tick",this.dc,!1,this);zc(this,this.h)}
v(Pi,I);p=Pi.prototype;p.sendIsolatedPayload=function(a){this.u=a;this.o=1};
function Qi(a){a.h.enabled||a.h.start();a.j++;a.j>=a.o&&a.dc()}
p.dc=function(){var a=this.i.values();a=[].concat(A(a)).filter(function(b){return b.h.size});
a.length&&this.G.flush(a,this.u);Ri(a);this.j=0;this.h.enabled&&this.h.stop()};
p.Nb=function(a){var b=C.apply(1,arguments);this.i.has(a)||this.i.set(a,new Gh(a,b))};
p.Jc=function(a){var b=C.apply(1,arguments);this.i.has(a)||this.i.set(a,new Hh(a,b))};
function Si(a,b){return a.A.has(b)?void 0:a.i.get(b)}
p.Kb=function(a){this.Vd(a,1,C.apply(1,arguments))};
p.Vd=function(a,b){var c=C.apply(2,arguments),d=Si(this,a);d&&d instanceof Gh&&(d.j(b,c),Qi(this))};
p.record=function(a,b){var c=C.apply(2,arguments),d=Si(this,a);d&&d instanceof Hh&&(d.record(b,c),Qi(this))};
function Ri(a){for(var b=0;b<a.length;b++)a[b].clear()}
;function Ti(){}
Ti.prototype.serialize=function(a){var b=[];Ui(this,a,b);return b.join("")};
function Ui(a,b,c){if(b==null)c.push("null");else{if(typeof b=="object"){if(Array.isArray(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),Ui(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],typeof f!="function"&&(c.push(e),Vi(d,c),c.push(":"),Ui(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":Vi(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var Wi={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\v":"\\u000b"},Xi=/\uffff/.test("\uffff")?/[\\"\x00-\x1f\x7f-\uffff]/g:/[\\"\x00-\x1f\x7f-\xff]/g;function Vi(a,b){b.push('"',a.replace(Xi,function(c){var d=Wi[c];d||(d="\\u"+(c.charCodeAt(0)|65536).toString(16).slice(1),Wi[c]=d);return d}),'"')}
;function Yi(){fi.call(this);this.headers=new Map;this.h=!1;this.K=null;this.o=this.Y="";this.j=this.U=this.A=this.P=!1;this.G=0;this.u=null;this.ma="";this.ha=!1}
$a(Yi,fi);var Zi=/^https?$/i,$i=["POST","PUT"],aj=[];function bj(a,b,c,d,e,f,g){var h=new Yi;aj.push(h);b&&h.listen("complete",b);bi(h,"ready",h.ee);f&&(h.G=Math.max(0,f));g&&(h.ha=g);h.send(a,c,d,e)}
p=Yi.prototype;p.ee=function(){this.dispose();Yb(aj,this)};
p.send=function(a,b,c,d){if(this.K)throw Error("[goog.net.XhrIo] Object is active with another request="+this.Y+"; newUri="+a);b=b?b.toUpperCase():"GET";this.Y=a;this.o="";this.P=!1;this.h=!0;this.K=new XMLHttpRequest;this.K.onreadystatechange=ii(Ua(this.Dd,this));try{this.getStatus(),this.U=!0,this.K.open(b,String(a),!0),this.U=!1}catch(g){this.getStatus();cj(this,g);return}a=c||"";c=new Map(this.headers);if(d)if(Object.getPrototypeOf(d)===Object.prototype)for(var e in d)c.set(e,d[e]);else if(typeof d.keys===
"function"&&typeof d.get==="function"){e=y(d.keys());for(var f=e.next();!f.done;f=e.next())f=f.value,c.set(f,d.get(f))}else throw Error("Unknown input type for opt_headers: "+String(d));d=Array.from(c.keys()).find(function(g){return"content-type"==g.toLowerCase()});
e=D.FormData&&a instanceof D.FormData;!(Sb($i,b)>=0)||d||e||c.set("Content-Type","application/x-www-form-urlencoded;charset=utf-8");b=y(c);for(d=b.next();!d.done;d=b.next())c=y(d.value),d=c.next().value,c=c.next().value,this.K.setRequestHeader(d,c);this.ma&&(this.K.responseType=this.ma);"withCredentials"in this.K&&this.K.withCredentials!==this.ha&&(this.K.withCredentials=this.ha);try{this.u&&(clearTimeout(this.u),this.u=null),this.G>0&&(this.getStatus(),this.u=setTimeout(this.xf.bind(this),this.G)),
this.getStatus(),this.A=!0,this.K.send(a),this.A=!1}catch(g){this.getStatus(),cj(this,g)}};
p.xf=function(){typeof Ha!="undefined"&&this.K&&(this.o="Timed out after "+this.G+"ms, aborting",this.getStatus(),gi(this,"timeout"),this.abort(8))};
function cj(a,b){a.h=!1;a.K&&(a.j=!0,a.K.abort(),a.j=!1);a.o=b;dj(a);ej(a)}
function dj(a){a.P||(a.P=!0,gi(a,"complete"),gi(a,"error"))}
p.abort=function(){this.K&&this.h&&(this.getStatus(),this.h=!1,this.j=!0,this.K.abort(),this.j=!1,gi(this,"complete"),gi(this,"abort"),ej(this))};
p.ba=function(){this.K&&(this.h&&(this.h=!1,this.j=!0,this.K.abort(),this.j=!1),ej(this,!0));Yi.Aa.ba.call(this)};
p.Dd=function(){this.ea||(this.U||this.A||this.j?fj(this):this.Oe())};
p.Oe=function(){fj(this)};
function fj(a){if(a.h&&typeof Ha!="undefined")if(a.A&&(a.K?a.K.readyState:0)==4)setTimeout(a.Dd.bind(a),0);else if(gi(a,"readystatechange"),a.isComplete()){a.getStatus();a.h=!1;try{if(gj(a))gi(a,"complete"),gi(a,"success");else{try{var b=(a.K?a.K.readyState:0)>2?a.K.statusText:""}catch(c){b=""}a.o=b+" ["+a.getStatus()+"]";dj(a)}}finally{ej(a)}}}
function ej(a,b){if(a.K){a.u&&(clearTimeout(a.u),a.u=null);var c=a.K;a.K=null;b||gi(a,"ready");try{c.onreadystatechange=null}catch(d){}}}
p.isActive=function(){return!!this.K};
p.isComplete=function(){return(this.K?this.K.readyState:0)==4};
function gj(a){var b=a.getStatus();a:switch(b){case 200:case 201:case 202:case 204:case 206:case 304:case 1223:var c=!0;break a;default:c=!1}if(!c){if(b=b===0)a=String(a.Y).match(ic)[1]||null,!a&&D.self&&D.self.location&&(a=D.self.location.protocol.slice(0,-1)),b=!Zi.test(a?a.toLowerCase():"");c=b}return c}
p.getStatus=function(){try{return(this.K?this.K.readyState:0)>2?this.K.status:-1}catch(a){return-1}};
p.getLastError=function(){return typeof this.o==="string"?this.o:String(this.o)};function hj(){}
hj.prototype.send=function(a,b,c){b=b===void 0?function(){}:b;
c=c===void 0?function(){}:c;
bj(a.url,function(d){d=d.target;if(gj(d)){try{var e=d.K?d.K.responseText:""}catch(f){e=""}b(e)}else c(d.getStatus())},a.requestType,a.body,a.Ec,a.timeoutMillis,a.withCredentials)};
hj.prototype.qc=function(){return 1};function ij(a,b){this.logger=a;this.event=b;this.startTime=jj()}
ij.prototype.done=function(){this.logger.Wb(this.event,jj()-this.startTime)};
function kj(){Kc.apply(this,arguments)}
v(kj,Kc);function lj(a,b){var c=jj();b=b();a.Wb("n",jj()-c);return b}
function mj(){kj.apply(this,arguments)}
v(mj,kj);p=mj.prototype;p.Rc=function(){};
p.Db=function(){};
p.Wb=function(){};
p.Ha=function(){};
p.Dc=function(){};
p.Pd=function(){};
function nj(a){return{sf:new Nc(a),errorCount:new Rc(a),eventCount:new Pc(a),pe:new Qc(a),oi:new Oc(a),si:new Sc(a),wh:new Tc(a),ri:new Uc(a)}}
function oj(a,b,c,d){a=zh(xh(wh(new vh(1828,"0"),a),new hj));b.length&&yh(a,Vf(new Uf,b));d!==void 0&&(a.Va=d);var e=new Bh(1828,"","",!1,"",Ah(a));zc(e,a);var f=new Pi({flush:function(g){try{e.flush(g)}catch(h){c(h)}}});
f.addOnDisposeCallback(function(){setTimeout(function(){try{f.dc()}finally{e.dispose()}})});
f.o=1E5;f.flushInterval=3E4;f.h.setInterval(3E4);return f}
function pj(a,b){I.call(this);var c=this;this.callback=a;this.i=b;this.h=-b;this.addOnDisposeCallback(function(){return void clearTimeout(c.timer)})}
v(pj,I);function qj(a){if(a.timer===void 0){var b=Math.max(0,a.h+a.i-jj());a.timer=setTimeout(function(){try{a.callback()}finally{a.h=jj(),a.timer=void 0}},b)}}
function rj(a,b){kj.call(this);this.metrics=a;this.Da=b}
v(rj,kj);rj.prototype.Rc=function(a){this.metrics.sf.record(a,this.Da)};
rj.prototype.Db=function(a){this.metrics.eventCount.kb(a,this.Da)};
rj.prototype.Wb=function(a,b){this.metrics.pe.record(b,a,this.Da)};
rj.prototype.Ha=function(a){this.metrics.errorCount.kb(a,this.Da)};
function sj(a,b){b=b===void 0?[]:b;var c={Da:a.Da||"_",pc:a.pc||[],xc:a.xc|0,Va:a.Va,yc:a.yc||function(){},
Jb:a.Jb||function(e,f){return oj(e,f,c.yc,c.Va)}};
b=c.Jb("49",c.pc.concat(b));rj.call(this,nj(b),c.Da);var d=this;this.options=c;this.service=b;this.i=!a.Jb;this.h=new pj(function(){return void d.service.dc()},c.xc);
this.addOnDisposeCallback(function(){d.h.dispose();d.i&&d.service.dispose()})}
v(sj,rj);sj.prototype.Pd=function(a){var b=this;this.h.dispose();this.i&&this.service.dispose();this.service=this.options.Jb("49",this.options.pc.concat(a));this.h=new pj(function(){return void b.service.dc()},this.options.xc);
this.metrics=nj(this.service)};
sj.prototype.Dc=function(){qj(this.h)};
function jj(){var a,b,c;return(c=(a=globalThis.performance)==null?void 0:(b=a.now)==null?void 0:b.call(a))!=null?c:Date.now()}
;function tj(a){this.F=L(a)}
v(tj,M);function uj(a){this.F=L(a)}
v(uj,M);function vj(a){this.F=L(a,0,"bfkj")}
v(vj,M);var wj=function(a){return he(function(b){return b instanceof a&&!de(b)})}(vj);
vj.Me="bfkj";function Gc(a){this.F=L(a)}
v(Gc,M);function xj(a){this.F=L(a)}
v(xj,M);var yj=Tf(xj);function zj(){var a=this;this.promise=new Promise(function(b,c){a.resolve=b;a.reject=c})}
;function Aj(a,b,c){if(a.disable)return new mj;b=b?Ec(b):[];if(c)return c.Pd(b),c.share();a={Da:a.Da,pc:a.Kh,xc:a.Wh,Va:a.Va,yc:a.yc,Jb:a.Jb};c=b;c=c===void 0?[]:c;return new sj(a,c)}
function Bj(a){function b(w,x,z,G){Promise.resolve().then(function(){k.done();h.Dc();h.dispose();g.resolve({Zd:w,rf:x,Se:z,yh:G})})}
function c(w,x,z,G){if(!d.logger.ea){var H="k";x?H="h":z&&(H="u");H!=="k"?G!==0&&(d.logger.Db(H),d.logger.Wb(H,w)):d.i<=0?(d.logger.Db(H),d.logger.Wb(H,w),d.i=Math.floor(Math.random()*200)):d.i--}}
I.call(this);var d=this;this.i=Math.floor(Math.random()*200);this.h=new xj;if("challenge"in a&&wj(a.challenge)){var e=Kf(a.challenge,4,void 0,ge);var f=Kf(a.challenge,5,void 0,ge);Kf(a.challenge,7,void 0,ge)&&(this.h=yj(Kf(a.challenge,7,void 0,ge)))}else e=a.program,f=a.globalName;this.addOnDisposeCallback(function(){var w,x,z;return B(function(G){if(G.h==1)return G.yield(d.j,2);w=G.i;x=w.rf;(z=x)==null||z();G.h=0})});
this.logger=Aj(a.Bd||{},this.h,a.zh);zc(this,this.logger);var g=new zj;this.j=g.promise;this.logger.Db("t");var h=this.logger.share(),k=new ij(h,"t");if(!D[f])throw this.logger.Ha(25),Error("EGOU");if(!D[f].a)throw this.logger.Ha(26),Error("ELIU");try{var l=D[f].a;f=[];for(var m=[],n=Ec(this.h),r=0;r<n.length;r++)f.push(n[r]),m.push(1);var t=Ic(this.h);for(n=0;n<t.length;n++)f.push(t[n]),m.push(2);this.u=y(l(e,b,!0,a.ni,c,[f,m],Kf(this.h,5))).next().value;this.cd=g.promise.then(function(){})}catch(w){throw this.logger.Ha(28),
w;
}}
v(Bj,I);Bj.prototype.snapshot=function(a){if(this.ea)throw Error("Already disposed");this.logger.Db("n");var b=this.logger.share();return this.j.then(function(c){var d=c.Zd;return new Promise(function(e){var f=new ij(b,"n");d(function(g){f.done();b.Rc(g.length);b.Dc();b.dispose();e(g)},[a.wb,
a.ed,a.Cf,a.gd])})})};
Bj.prototype.hd=function(a){var b=this;if(this.ea)throw Error("Already disposed");this.logger.Db("n");var c=lj(this.logger,function(){return b.u([a.wb,a.ed,a.Cf,a.gd])});
this.logger.Rc(c.length);this.logger.Dc();return c};
Bj.prototype.o=function(a){this.j.then(function(b){var c;(c=b.Se)==null||c(a)})};function Cj(a){if(!a)return null;a=af(vf(a,4,void 0,uf));return a===null||a===void 0?null:ob(a)}
;function Dj(){this.promises={};this.h=null}
function Ej(){Dj.instance||(Dj.instance=new Dj);return Dj.instance}
function Fj(a,b){return Gj(a,Gf(b,tj,1,ge),Gf(b,uj,2,ge),Kf(b,3,void 0,ge))}
function Gj(a,b,c,d){if(!b&&!c)return Promise.resolve();if(!d)return Hj(b,c);var e;(e=a.promises)[d]||(e[d]=new Promise(function(f,g){Hj(b,c).then(function(){a.h=d;f()},function(h){delete a.promises[d];
g(h)})}));
return a.promises[d]}
function Hj(a,b){return b?Ij(b):a?Jj(a):Promise.resolve()}
function Ij(a){return new Promise(function(b,c){var d=Jg("SCRIPT"),e=Cj(a);Lb(d,e);d.onload=function(){Kg(d);b()};
d.onerror=function(){Kg(d);c(Error("EWLS"))};
(document.getElementsByTagName("HEAD")[0]||document.documentElement).appendChild(d)})}
function Jj(a){return new Promise(function(b){var c=Jg("SCRIPT");if(a){var d=af(vf(a,6,void 0,uf));d=d===null||d===void 0?null:Ib(d)}else d=null;c.textContent=Jb(d);Kb(c);(document.getElementsByTagName("HEAD")[0]||document.documentElement).appendChild(c);Kg(c);b()})}
;var Kj=window;function Lj(a){var b=Mj;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a(b[c],c,b)}
function Nj(){var a=[];Lj(function(b){a.push(b)});
return a}
var Mj={Df:"allow-forms",Ef:"allow-modals",Ff:"allow-orientation-lock",Gf:"allow-pointer-lock",Hf:"allow-popups",If:"allow-popups-to-escape-sandbox",Jf:"allow-presentation",Kf:"allow-same-origin",Lf:"allow-scripts",Mf:"allow-top-navigation",Nf:"allow-top-navigation-by-user-activation"},Oj=vi(function(){return Nj()});
function Pj(){var a=Qj(),b={};Tb(Oj(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
function Qj(){var a=a===void 0?document:a;return a.createElement("iframe")}
;function Rj(a){typeof a=="number"&&(a=Math.round(a)+"px");return a}
;var Sj=(new Date).getTime();Object.assign({},{attributes:{},handleError:function(a){throw a;}},{Dh:!0,
Fh:!0,Gh:Xc,Hh:Yc,Ih:!1,uh:!1,Eh:!0,Ch:!1});function Tj(a){fi.call(this);var b=this;this.A=this.j=0;this.Ca=a!=null?a:{pa:function(e,f){return setTimeout(e,f)},
qa:function(e){clearTimeout(e)}};
var c,d;this.h=(d=(c=window.navigator)==null?void 0:c.onLine)!=null?d:!0;this.o=function(){return B(function(e){return e.yield(Uj(b),0)})};
window.addEventListener("offline",this.o);window.addEventListener("online",this.o);this.A||Vj(this)}
v(Tj,fi);function Wj(){var a=Xj;Tj.instance||(Tj.instance=new Tj(a));return Tj.instance}
Tj.prototype.dispose=function(){window.removeEventListener("offline",this.o);window.removeEventListener("online",this.o);this.Ca.qa(this.A);delete Tj.instance};
Tj.prototype.ta=function(){return this.h};
function Vj(a){a.A=a.Ca.pa(function(){var b;return B(function(c){if(c.h==1)return a.h?((b=window.navigator)==null?0:b.onLine)?c.B(3):c.yield(Uj(a),3):c.yield(Uj(a),3);Vj(a);c.h=0})},3E4)}
function Uj(a,b){return a.u?a.u:a.u=new Promise(function(c){var d,e,f,g;return B(function(h){switch(h.h){case 1:return d=window.AbortController?new window.AbortController:void 0,f=(e=d)==null?void 0:e.signal,g=!1,va(h,2,3),d&&(a.j=a.Ca.pa(function(){d.abort()},b||2E4)),h.yield(fetch("/generate_204",{method:"HEAD",
signal:f}),5);case 5:g=!0;case 3:h.P=[h.j];h.H=0;h.o=0;a.u=void 0;a.j&&(a.Ca.qa(a.j),a.j=0);g!==a.h&&(a.h=g,a.h?gi(a,"networkstatus-online"):gi(a,"networkstatus-offline"));c(g);ya(h);break;case 2:xa(h),g=!1,h.B(3)}})})}
;function Yj(){this.data=[];this.h=-1}
Yj.prototype.set=function(a,b){b=b===void 0?!0:b;0<=a&&a<52&&Number.isInteger(a)&&this.data[a]!==b&&(this.data[a]=b,this.h=-1)};
Yj.prototype.get=function(a){return!!this.data[a]};
function Zj(a){a.h===-1&&(a.h=a.data.reduce(function(b,c,d){return b+(c?Math.pow(2,d):0)},0));
return a.h}
;function ak(){this.blockSize=-1}
;function bk(){this.blockSize=-1;this.blockSize=64;this.h=[];this.u=[];this.H=[];this.j=[];this.j[0]=128;for(var a=1;a<this.blockSize;++a)this.j[a]=0;this.o=this.i=0;this.reset()}
$a(bk,ak);bk.prototype.reset=function(){this.h[0]=1732584193;this.h[1]=4023233417;this.h[2]=2562383102;this.h[3]=271733878;this.h[4]=3285377520;this.o=this.i=0};
function ck(a,b,c){c||(c=0);var d=a.H;if(typeof b==="string")for(var e=0;e<16;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;e<16;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(b=16;b<80;b++)c=d[b-3]^d[b-8]^d[b-14]^d[b-16],d[b]=(c<<1|c>>>31)&4294967295;b=a.h[0];c=a.h[1];e=a.h[2];for(var f=a.h[3],g=a.h[4],h,k,l=0;l<80;l++)l<40?l<20?(h=f^c&(e^f),k=1518500249):(h=c^e^f,k=1859775393):l<60?(h=c&e|f&(c|e),k=2400959708):(h=c^e^f,k=3395469782),
h=(b<<5|b>>>27)+h+g+k+d[l]&4294967295,g=f,f=e,e=(c<<30|c>>>2)&4294967295,c=b,b=h;a.h[0]=a.h[0]+b&4294967295;a.h[1]=a.h[1]+c&4294967295;a.h[2]=a.h[2]+e&4294967295;a.h[3]=a.h[3]+f&4294967295;a.h[4]=a.h[4]+g&4294967295}
bk.prototype.update=function(a,b){if(a!=null){b===void 0&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.u,f=this.i;d<b;){if(f==0)for(;d<=c;)ck(this,a,d),d+=this.blockSize;if(typeof a==="string")for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){ck(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){ck(this,e);f=0;break}}this.i=f;this.o+=b}};
bk.prototype.digest=function(){var a=[],b=this.o*8;this.i<56?this.update(this.j,56-this.i):this.update(this.j,this.blockSize-(this.i-56));for(var c=this.blockSize-1;c>=56;c--)this.u[c]=b&255,b/=256;ck(this,this.u);for(c=b=0;c<5;c++)for(var d=24;d>=0;d-=8)a[b]=this.h[c]>>d&255,++b;return a};function dk(a){return typeof a.className=="string"?a.className:a.getAttribute&&a.getAttribute("class")||""}
function ek(a,b){typeof a.className=="string"?a.className=b:a.setAttribute&&a.setAttribute("class",b)}
function fk(a,b){a.classList?b=a.classList.contains(b):(a=a.classList?a.classList:dk(a).match(/\S+/g)||[],b=Sb(a,b)>=0);return b}
function gk(){var a=document.body;a.classList?a.classList.remove("inverted-hdpi"):fk(a,"inverted-hdpi")&&ek(a,Array.prototype.filter.call(a.classList?a.classList:dk(a).match(/\S+/g)||[],function(b){return b!="inverted-hdpi"}).join(" "))}
;function hk(){}
hk.prototype.next=function(){return ik};
var ik={done:!0,value:void 0};function jk(a){return{value:a,done:!1}}
hk.prototype.tb=function(){return this};function kk(a){if(a instanceof lk||a instanceof mk||a instanceof nk)return a;if(typeof a.next=="function")return new lk(function(){return a});
if(typeof a[Symbol.iterator]=="function")return new lk(function(){return a[Symbol.iterator]()});
if(typeof a.tb=="function")return new lk(function(){return a.tb()});
throw Error("Not an iterator or iterable.");}
function lk(a){this.h=a}
lk.prototype.tb=function(){return new mk(this.h())};
lk.prototype[Symbol.iterator]=function(){return new nk(this.h())};
lk.prototype.i=function(){return new nk(this.h())};
function mk(a){this.h=a}
v(mk,hk);mk.prototype.next=function(){return this.h.next()};
mk.prototype[Symbol.iterator]=function(){return new nk(this.h)};
mk.prototype.i=function(){return new nk(this.h)};
function nk(a){lk.call(this,function(){return a});
this.j=a}
v(nk,lk);nk.prototype.next=function(){return this.j.next()};function N(a){I.call(this);this.u=1;this.j=[];this.o=0;this.h=[];this.i={};this.A=!!a}
$a(N,I);p=N.prototype;p.subscribe=function(a,b,c){var d=this.i[a];d||(d=this.i[a]=[]);var e=this.u;this.h[e]=a;this.h[e+1]=b;this.h[e+2]=c;this.u=e+3;d.push(e);return e};
p.unsubscribe=function(a,b,c){if(a=this.i[a]){var d=this.h;if(a=a.find(function(e){return d[e+1]==b&&d[e+2]==c}))return this.fc(a)}return!1};
p.fc=function(a){var b=this.h[a];if(b){var c=this.i[b];this.o!=0?(this.j.push(a),this.h[a+1]=function(){}):(c&&Yb(c,a),delete this.h[a],delete this.h[a+1],delete this.h[a+2])}return!!b};
p.sb=function(a,b){var c=this.i[a];if(c){var d=Array(arguments.length-1),e=arguments.length,f;for(f=1;f<e;f++)d[f-1]=arguments[f];if(this.A)for(f=0;f<c.length;f++)e=c[f],ok(this.h[e+1],this.h[e+2],d);else{this.o++;try{for(f=0,e=c.length;f<e&&!this.ea;f++){var g=c[f];this.h[g+1].apply(this.h[g+2],d)}}finally{if(this.o--,this.j.length>0&&this.o==0)for(;c=this.j.pop();)this.fc(c)}}return f!=0}return!1};
function ok(a,b,c){ri(function(){a.apply(b,c)})}
p.clear=function(a){if(a){var b=this.i[a];b&&(b.forEach(this.fc,this),delete this.i[a])}else this.h.length=0,this.i={}};
p.ba=function(){N.Aa.ba.call(this);this.clear();this.j.length=0};function pk(a){this.h=a}
pk.prototype.set=function(a,b){b===void 0?this.h.remove(a):this.h.set(a,(new Ti).serialize(b))};
pk.prototype.get=function(a){try{var b=this.h.get(a)}catch(c){return}if(b!==null)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
pk.prototype.remove=function(a){this.h.remove(a)};function qk(a){this.h=a}
$a(qk,pk);function rk(a){this.data=a}
function sk(a){return a===void 0||a instanceof rk?a:new rk(a)}
qk.prototype.set=function(a,b){qk.Aa.set.call(this,a,sk(b))};
qk.prototype.i=function(a){a=qk.Aa.get.call(this,a);if(a===void 0||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
qk.prototype.get=function(a){if(a=this.i(a)){if(a=a.data,a===void 0)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function tk(a){this.h=a}
$a(tk,qk);tk.prototype.set=function(a,b,c){if(b=sk(b)){if(c){if(c<Ya()){tk.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Ya()}tk.Aa.set.call(this,a,b)};
tk.prototype.i=function(a){var b=tk.Aa.i.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Ya()||c&&c>Ya())tk.prototype.remove.call(this,a);else return b}};function uk(){}
;function vk(){}
$a(vk,uk);vk.prototype[Symbol.iterator]=function(){return kk(this.tb(!0)).i()};
vk.prototype.clear=function(){var a=Array.from(this);a=y(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function wk(a){this.h=a;this.i=null}
$a(wk,vk);p=wk.prototype;p.isAvailable=function(){if(!Zc||this.i===null){var a=this.h;if(a)try{performance.now();a.setItem("__sak","1");a.removeItem("__sak");performance.now();var b=!0}catch(c){b=c instanceof DOMException&&(c.name==="QuotaExceededError"||c.code===22||c.code===1014||c.name==="NS_ERROR_DOM_QUOTA_REACHED")&&a&&a.length!==0}else b=!1;this.i=b}return this.i};
p.set=function(a,b){xk(this);try{this.h.setItem(a,b)}catch(c){if(this.h.length==0)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
p.get=function(a){xk(this);a=this.h.getItem(a);if(typeof a!=="string"&&a!==null)throw"Storage mechanism: Invalid value was encountered";return a};
p.remove=function(a){xk(this);this.h.removeItem(a)};
p.tb=function(a){xk(this);var b=0,c=this.h,d=new hk;d.next=function(){if(b>=c.length)return ik;var e=c.key(b++);if(a)return jk(e);e=c.getItem(e);if(typeof e!=="string")throw"Storage mechanism: Invalid value was encountered";return jk(e)};
return d};
p.clear=function(){xk(this);this.h.clear()};
p.key=function(a){xk(this);return this.h.key(a)};
function xk(a){if(a.h==null)throw Error("Storage mechanism: Storage unavailable");var b;(Zc?a.isAvailable():(b=a.i)!=null?b:a.isAvailable())||Vc(Error("Storage mechanism: Storage unavailable"))}
;function yk(){var a=null;try{a=D.localStorage||null}catch(b){}wk.call(this,a)}
$a(yk,wk);function zk(a,b){this.i=a;this.h=b+"::"}
$a(zk,vk);zk.prototype.set=function(a,b){this.i.set(this.h+a,b)};
zk.prototype.get=function(a){return this.i.get(this.h+a)};
zk.prototype.remove=function(a){this.i.remove(this.h+a)};
zk.prototype.tb=function(a){var b=this.i[Symbol.iterator](),c=this,d=new hk;d.next=function(){var e=b.next();if(e.done)return e;for(e=e.value;e.slice(0,c.h.length)!=c.h;){e=b.next();if(e.done)return e;e=e.value}return jk(a?e.slice(c.h.length):c.i.get(e))};
return d};function Ak(a){if(a.Wa&&typeof a.Wa=="function")return a.Wa();if(typeof Map!=="undefined"&&a instanceof Map||typeof Set!=="undefined"&&a instanceof Set)return Array.from(a.values());if(typeof a==="string")return a.split("");if(Ma(a)){for(var b=[],c=a.length,d=0;d<c;d++)b.push(a[d]);return b}return vg(a)}
function Bk(a){if(a.Tb&&typeof a.Tb=="function")return a.Tb();if(!a.Wa||typeof a.Wa!="function"){if(typeof Map!=="undefined"&&a instanceof Map)return Array.from(a.keys());if(!(typeof Set!=="undefined"&&a instanceof Set)){if(Ma(a)||typeof a==="string"){var b=[];a=a.length;for(var c=0;c<a;c++)b.push(c);return b}b=[];c=0;for(var d in a)b[c++]=d;return b}}}
function Ck(a,b,c){if(a.forEach&&typeof a.forEach=="function")a.forEach(b,c);else if(Ma(a)||typeof a==="string")Array.prototype.forEach.call(a,b,c);else for(var d=Bk(a),e=Ak(a),f=e.length,g=0;g<f;g++)b.call(c,e[g],d&&d[g],a)}
;function Dk(a){this.i=this.A=this.j="";this.G=null;this.u=this.h="";this.o=!1;var b;a instanceof Dk?(this.o=a.o,Ek(this,a.j),this.A=a.A,this.i=a.i,Fk(this,a.G),this.h=a.h,Gk(this,a.H.clone()),this.u=a.u):a&&(b=String(a).match(ic))?(this.o=!1,Ek(this,b[1]||"",!0),this.A=Hk(b[2]||""),this.i=Hk(b[3]||"",!0),Fk(this,b[4]),this.h=Hk(b[5]||"",!0),Gk(this,b[6]||"",!0),this.u=Hk(b[7]||"")):(this.o=!1,this.H=new Ik(null,this.o))}
Dk.prototype.toString=function(){var a=[],b=this.j;b&&a.push(Jk(b,Kk,!0),":");var c=this.i;if(c||b=="file")a.push("//"),(b=this.A)&&a.push(Jk(b,Kk,!0),"@"),a.push(Lk(encodeURIComponent(String(c)))),c=this.G,c!=null&&a.push(":",String(c));if(c=this.h)this.i&&c.charAt(0)!="/"&&a.push("/"),a.push(Jk(c,c.charAt(0)=="/"?Mk:Nk,!0));(c=this.H.toString())&&a.push("?",c);(c=this.u)&&a.push("#",Jk(c,Ok));return a.join("")};
Dk.prototype.resolve=function(a){var b=this.clone(),c=!!a.j;c?Ek(b,a.j):c=!!a.A;c?b.A=a.A:c=!!a.i;c?b.i=a.i:c=a.G!=null;var d=a.h;if(c)Fk(b,a.G);else if(c=!!a.h){if(d.charAt(0)!="/")if(this.i&&!this.h)d="/"+d;else{var e=b.h.lastIndexOf("/");e!=-1&&(d=b.h.slice(0,e+1)+d)}e=d;if(e==".."||e==".")d="";else if(e.indexOf("./")!=-1||e.indexOf("/.")!=-1){d=e.lastIndexOf("/",0)==0;e=e.split("/");for(var f=[],g=0;g<e.length;){var h=e[g++];h=="."?d&&g==e.length&&f.push(""):h==".."?((f.length>1||f.length==1&&
f[0]!="")&&f.pop(),d&&g==e.length&&f.push("")):(f.push(h),d=!0)}d=f.join("/")}else d=e}c?b.h=d:c=a.H.toString()!=="";c?Gk(b,a.H.clone()):c=!!a.u;c&&(b.u=a.u);return b};
Dk.prototype.clone=function(){return new Dk(this)};
function Ek(a,b,c){a.j=c?Hk(b,!0):b;a.j&&(a.j=a.j.replace(/:$/,""))}
function Fk(a,b){if(b){b=Number(b);if(isNaN(b)||b<0)throw Error("Bad port number "+b);a.G=b}else a.G=null}
function Gk(a,b,c){b instanceof Ik?(a.H=b,Pk(a.H,a.o)):(c||(b=Jk(b,Qk)),a.H=new Ik(b,a.o))}
function Hk(a,b){return a?b?decodeURI(a.replace(/%25/g,"%2525")):decodeURIComponent(a):""}
function Jk(a,b,c){return typeof a==="string"?(a=encodeURI(a).replace(b,Rk),c&&(a=Lk(a)),a):null}
function Rk(a){a=a.charCodeAt(0);return"%"+(a>>4&15).toString(16)+(a&15).toString(16)}
function Lk(a){return a.replace(/%25([0-9a-fA-F]{2})/g,"%$1")}
var Kk=/[#\/\?@]/g,Nk=/[#\?:]/g,Mk=/[#\?]/g,Qk=/[#\?@]/g,Ok=/#/g;function Ik(a,b){this.i=this.h=null;this.j=a||null;this.o=!!b}
function Sk(a){a.h||(a.h=new Map,a.i=0,a.j&&oc(a.j,function(b,c){a.add(fc(b),c)}))}
p=Ik.prototype;p.add=function(a,b){Sk(this);this.j=null;a=Tk(this,a);var c=this.h.get(a);c||this.h.set(a,c=[]);c.push(b);this.i=this.i+1;return this};
p.remove=function(a){Sk(this);a=Tk(this,a);return this.h.has(a)?(this.j=null,this.i=this.i-this.h.get(a).length,this.h.delete(a)):!1};
p.clear=function(){this.h=this.j=null;this.i=0};
function Uk(a,b){Sk(a);b=Tk(a,b);return a.h.has(b)}
p.forEach=function(a,b){Sk(this);this.h.forEach(function(c,d){c.forEach(function(e){a.call(b,e,d,this)},this)},this)};
p.Tb=function(){Sk(this);for(var a=Array.from(this.h.values()),b=Array.from(this.h.keys()),c=[],d=0;d<b.length;d++)for(var e=a[d],f=0;f<e.length;f++)c.push(b[d]);return c};
p.Wa=function(a){Sk(this);var b=[];if(typeof a==="string")Uk(this,a)&&(b=b.concat(this.h.get(Tk(this,a))));else{a=Array.from(this.h.values());for(var c=0;c<a.length;c++)b=b.concat(a[c])}return b};
p.set=function(a,b){Sk(this);this.j=null;a=Tk(this,a);Uk(this,a)&&(this.i=this.i-this.h.get(a).length);this.h.set(a,[b]);this.i=this.i+1;return this};
p.get=function(a,b){if(!a)return b;a=this.Wa(a);return a.length>0?String(a[0]):b};
p.toString=function(){if(this.j)return this.j;if(!this.h)return"";for(var a=[],b=Array.from(this.h.keys()),c=0;c<b.length;c++){var d=b[c],e=encodeURIComponent(String(d));d=this.Wa(d);for(var f=0;f<d.length;f++){var g=e;d[f]!==""&&(g+="="+encodeURIComponent(String(d[f])));a.push(g)}}return this.j=a.join("&")};
p.clone=function(){var a=new Ik;a.j=this.j;this.h&&(a.h=new Map(this.h),a.i=this.i);return a};
function Tk(a,b){b=String(b);a.o&&(b=b.toLowerCase());return b}
function Pk(a,b){b&&!a.o&&(Sk(a),a.j=null,a.h.forEach(function(c,d){var e=d.toLowerCase();d!=e&&(this.remove(d),this.remove(e),c.length>0&&(this.j=null,this.h.set(Tk(this,e),Zb(c)),this.i=this.i+c.length))},a));
a.o=b}
p.extend=function(a){for(var b=0;b<arguments.length;b++)Ck(arguments[b],function(c,d){this.add(d,c)},this)};/*

 (The MIT License)

 Copyright (C) 2014 by Vitaly Puzrin

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

 -----------------------------------------------------------------------------
 Ported from zlib, which is under the following license
 https://github.com/madler/zlib/blob/master/zlib.h

 zlib.h -- interface of the 'zlib' general purpose compression library
   version 1.2.8, April 28th, 2013
   Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
   Jean-loup Gailly        Mark Adler
   jloup@gzip.org          madler@alumni.caltech.edu
   The data format used by the zlib library is described by RFCs (Request for
   Comments) 1950 to 1952 in the files http://tools.ietf.org/html/rfc1950
   (zlib format), rfc1951 (deflate format) and rfc1952 (gzip format).
*/
var O={},Vk=typeof Uint8Array!=="undefined"&&typeof Uint16Array!=="undefined"&&typeof Int32Array!=="undefined";O.assign=function(a){for(var b=Array.prototype.slice.call(arguments,1);b.length;){var c=b.shift();if(c){if(typeof c!=="object")throw new TypeError(c+"must be non-object");for(var d in c)Object.prototype.hasOwnProperty.call(c,d)&&(a[d]=c[d])}}return a};
O.dd=function(a,b){if(a.length===b)return a;if(a.subarray)return a.subarray(0,b);a.length=b;return a};
var Wk={ub:function(a,b,c,d,e){if(b.subarray&&a.subarray)a.set(b.subarray(c,c+d),e);else for(var f=0;f<d;f++)a[e+f]=b[c+f]},
ud:function(a){var b,c;var d=c=0;for(b=a.length;d<b;d++)c+=a[d].length;var e=new Uint8Array(c);d=c=0;for(b=a.length;d<b;d++){var f=a[d];e.set(f,c);c+=f.length}return e}},Xk={ub:function(a,b,c,d,e){for(var f=0;f<d;f++)a[e+f]=b[c+f]},
ud:function(a){return[].concat.apply([],a)}};
O.qf=function(){Vk?(O.rb=Uint8Array,O.Ja=Uint16Array,O.Ud=Int32Array,O.assign(O,Wk)):(O.rb=Array,O.Ja=Array,O.Ud=Array,O.assign(O,Xk))};
O.qf();var Yk=!0;try{new Uint8Array(1)}catch(a){Yk=!1}
function Zk(a){var b,c,d=a.length,e=0;for(b=0;b<d;b++){var f=a.charCodeAt(b);if((f&64512)===55296&&b+1<d){var g=a.charCodeAt(b+1);(g&64512)===56320&&(f=65536+(f-55296<<10)+(g-56320),b++)}e+=f<128?1:f<2048?2:f<65536?3:4}var h=new O.rb(e);for(b=c=0;c<e;b++)f=a.charCodeAt(b),(f&64512)===55296&&b+1<d&&(g=a.charCodeAt(b+1),(g&64512)===56320&&(f=65536+(f-55296<<10)+(g-56320),b++)),f<128?h[c++]=f:(f<2048?h[c++]=192|f>>>6:(f<65536?h[c++]=224|f>>>12:(h[c++]=240|f>>>18,h[c++]=128|f>>>12&63),h[c++]=128|f>>>
6&63),h[c++]=128|f&63);return h}
;var $k={};$k=function(a,b,c,d){var e=a&65535|0;a=a>>>16&65535|0;for(var f;c!==0;){f=c>2E3?2E3:c;c-=f;do e=e+b[d++]|0,a=a+e|0;while(--f);e%=65521;a%=65521}return e|a<<16|0};for(var al={},bl,cl=[],dl=0;dl<256;dl++){bl=dl;for(var el=0;el<8;el++)bl=bl&1?3988292384^bl>>>1:bl>>>1;cl[dl]=bl}al=function(a,b,c,d){c=d+c;for(a^=-1;d<c;d++)a=a>>>8^cl[(a^b[d])&255];return a^-1};var fl={};fl={2:"need dictionary",1:"stream end",0:"","-1":"file error","-2":"stream error","-3":"data error","-4":"insufficient memory","-5":"buffer error","-6":"incompatible version"};function gl(a){for(var b=a.length;--b>=0;)a[b]=0}
var hl=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0],il=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13],jl=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7],kl=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15],ll=Array(576);gl(ll);var ml=Array(60);gl(ml);var nl=Array(512);gl(nl);var ol=Array(256);gl(ol);var pl=Array(29);gl(pl);var ql=Array(30);gl(ql);function rl(a,b,c,d,e){this.Md=a;this.te=b;this.se=c;this.le=d;this.Le=e;this.xd=a&&a.length}
var sl,tl,ul;function vl(a,b){this.td=a;this.Fb=0;this.cb=b}
function wl(a,b){a.aa[a.pending++]=b&255;a.aa[a.pending++]=b>>>8&255}
function xl(a,b,c){a.ia>16-c?(a.oa|=b<<a.ia&65535,wl(a,a.oa),a.oa=b>>16-a.ia,a.ia+=c-16):(a.oa|=b<<a.ia&65535,a.ia+=c)}
function yl(a,b,c){xl(a,c[b*2],c[b*2+1])}
function zl(a,b){var c=0;do c|=a&1,a>>>=1,c<<=1;while(--b>0);return c>>>1}
function Al(a,b,c){var d=Array(16),e=0,f;for(f=1;f<=15;f++)d[f]=e=e+c[f-1]<<1;for(c=0;c<=b;c++)e=a[c*2+1],e!==0&&(a[c*2]=zl(d[e]++,e))}
function Bl(a){var b;for(b=0;b<286;b++)a.ra[b*2]=0;for(b=0;b<30;b++)a.hb[b*2]=0;for(b=0;b<19;b++)a.ja[b*2]=0;a.ra[512]=1;a.Pa=a.Ib=0;a.ya=a.matches=0}
function Cl(a){a.ia>8?wl(a,a.oa):a.ia>0&&(a.aa[a.pending++]=a.oa);a.oa=0;a.ia=0}
function Dl(a,b,c){Cl(a);wl(a,c);wl(a,~c);O.ub(a.aa,a.window,b,c,a.pending);a.pending+=c}
function El(a,b,c,d){var e=b*2,f=c*2;return a[e]<a[f]||a[e]===a[f]&&d[b]<=d[c]}
function Fl(a,b,c){for(var d=a.da[c],e=c<<1;e<=a.Na;){e<a.Na&&El(b,a.da[e+1],a.da[e],a.depth)&&e++;if(El(b,d,a.da[e],a.depth))break;a.da[c]=a.da[e];c=e;e<<=1}a.da[c]=d}
function Gl(a,b,c){var d=0;if(a.ya!==0){do{var e=a.aa[a.Qb+d*2]<<8|a.aa[a.Qb+d*2+1];var f=a.aa[a.Qc+d];d++;if(e===0)yl(a,f,b);else{var g=ol[f];yl(a,g+256+1,b);var h=hl[g];h!==0&&(f-=pl[g],xl(a,f,h));e--;g=e<256?nl[e]:nl[256+(e>>>7)];yl(a,g,c);h=il[g];h!==0&&(e-=ql[g],xl(a,e,h))}}while(d<a.ya)}yl(a,256,b)}
function Hl(a,b){var c=b.td,d=b.cb.Md,e=b.cb.xd,f=b.cb.le,g,h=-1;a.Na=0;a.Ab=573;for(g=0;g<f;g++)c[g*2]!==0?(a.da[++a.Na]=h=g,a.depth[g]=0):c[g*2+1]=0;for(;a.Na<2;){var k=a.da[++a.Na]=h<2?++h:0;c[k*2]=1;a.depth[k]=0;a.Pa--;e&&(a.Ib-=d[k*2+1])}b.Fb=h;for(g=a.Na>>1;g>=1;g--)Fl(a,c,g);k=f;do g=a.da[1],a.da[1]=a.da[a.Na--],Fl(a,c,1),d=a.da[1],a.da[--a.Ab]=g,a.da[--a.Ab]=d,c[k*2]=c[g*2]+c[d*2],a.depth[k]=(a.depth[g]>=a.depth[d]?a.depth[g]:a.depth[d])+1,c[g*2+1]=c[d*2+1]=k,a.da[1]=k++,Fl(a,c,1);while(a.Na>=
2);a.da[--a.Ab]=a.da[1];g=b.td;k=b.Fb;d=b.cb.Md;e=b.cb.xd;f=b.cb.te;var l=b.cb.se,m=b.cb.Le,n,r=0;for(n=0;n<=15;n++)a.Ka[n]=0;g[a.da[a.Ab]*2+1]=0;for(b=a.Ab+1;b<573;b++){var t=a.da[b];n=g[g[t*2+1]*2+1]+1;n>m&&(n=m,r++);g[t*2+1]=n;if(!(t>k)){a.Ka[n]++;var w=0;t>=l&&(w=f[t-l]);var x=g[t*2];a.Pa+=x*(n+w);e&&(a.Ib+=x*(d[t*2+1]+w))}}if(r!==0){do{for(n=m-1;a.Ka[n]===0;)n--;a.Ka[n]--;a.Ka[n+1]+=2;a.Ka[m]--;r-=2}while(r>0);for(n=m;n!==0;n--)for(t=a.Ka[n];t!==0;)d=a.da[--b],d>k||(g[d*2+1]!==n&&(a.Pa+=(n-g[d*
2+1])*g[d*2],g[d*2+1]=n),t--)}Al(c,h,a.Ka)}
function Il(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;f===0&&(h=138,k=3);b[(c+1)*2+1]=65535;for(d=0;d<=c;d++){var l=f;f=b[(d+1)*2+1];++g<h&&l===f||(g<k?a.ja[l*2]+=g:l!==0?(l!==e&&a.ja[l*2]++,a.ja[32]++):g<=10?a.ja[34]++:a.ja[36]++,g=0,e=l,f===0?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4))}}
function Jl(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;f===0&&(h=138,k=3);for(d=0;d<=c;d++){var l=f;f=b[(d+1)*2+1];if(!(++g<h&&l===f)){if(g<k){do yl(a,l,a.ja);while(--g!==0)}else l!==0?(l!==e&&(yl(a,l,a.ja),g--),yl(a,16,a.ja),xl(a,g-3,2)):g<=10?(yl(a,17,a.ja),xl(a,g-3,3)):(yl(a,18,a.ja),xl(a,g-11,7));g=0;e=l;f===0?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4)}}}
function Kl(a){var b=4093624447,c;for(c=0;c<=31;c++,b>>>=1)if(b&1&&a.ra[c*2]!==0)return 0;if(a.ra[18]!==0||a.ra[20]!==0||a.ra[26]!==0)return 1;for(c=32;c<256;c++)if(a.ra[c*2]!==0)return 1;return 0}
var Ll=!1;function Ml(a,b,c){a.aa[a.Qb+a.ya*2]=b>>>8&255;a.aa[a.Qb+a.ya*2+1]=b&255;a.aa[a.Qc+a.ya]=c&255;a.ya++;b===0?a.ra[c*2]++:(a.matches++,b--,a.ra[(ol[c]+256+1)*2]++,a.hb[(b<256?nl[b]:nl[256+(b>>>7)])*2]++);return a.ya===a.Vb-1}
;function Nl(a,b){a.msg=fl[b];return b}
function Ol(a){for(var b=a.length;--b>=0;)a[b]=0}
function Pl(a){var b=a.state,c=b.pending;c>a.S&&(c=a.S);c!==0&&(O.ub(a.output,b.aa,b.Yb,c,a.Gb),a.Gb+=c,b.Yb+=c,a.jd+=c,a.S-=c,b.pending-=c,b.pending===0&&(b.Yb=0))}
function Ql(a,b){var c=a.va>=0?a.va:-1,d=a.v-a.va,e=0;if(a.level>0){a.M.Lc===2&&(a.M.Lc=Kl(a));Hl(a,a.wc);Hl(a,a.nc);Il(a,a.ra,a.wc.Fb);Il(a,a.hb,a.nc.Fb);Hl(a,a.qd);for(e=18;e>=3&&a.ja[kl[e]*2+1]===0;e--);a.Pa+=3*(e+1)+5+5+4;var f=a.Pa+3+7>>>3;var g=a.Ib+3+7>>>3;g<=f&&(f=g)}else f=g=d+5;if(d+4<=f&&c!==-1)xl(a,b?1:0,3),Dl(a,c,d);else if(a.strategy===4||g===f)xl(a,2+(b?1:0),3),Gl(a,ll,ml);else{xl(a,4+(b?1:0),3);c=a.wc.Fb+1;d=a.nc.Fb+1;e+=1;xl(a,c-257,5);xl(a,d-1,5);xl(a,e-4,4);for(f=0;f<e;f++)xl(a,
a.ja[kl[f]*2+1],3);Jl(a,a.ra,c-1);Jl(a,a.hb,d-1);Gl(a,a.ra,a.hb)}Bl(a);b&&Cl(a);a.va=a.v;Pl(a.M)}
function P(a,b){a.aa[a.pending++]=b}
function Rl(a,b){a.aa[a.pending++]=b>>>8&255;a.aa[a.pending++]=b&255}
function Sl(a,b){var c=a.Ad,d=a.v,e=a.wa,f=a.Cd,g=a.v>a.la-262?a.v-(a.la-262):0,h=a.window,k=a.eb,l=a.Ia,m=a.v+258,n=h[d+e-1],r=h[d+e];a.wa>=a.wd&&(c>>=2);f>a.D&&(f=a.D);do{var t=b;if(h[t+e]===r&&h[t+e-1]===n&&h[t]===h[d]&&h[++t]===h[d+1]){d+=2;for(t++;h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&d<m;);t=258-(m-d);d=m-258;if(t>e){a.Eb=b;e=t;if(t>=f)break;n=h[d+e-1];r=h[d+e]}}}while((b=l[b&k])>g&&--c!==0);return e<=
a.D?e:a.D}
function Tl(a){var b=a.la,c;do{var d=a.Sd-a.D-a.v;if(a.v>=b+(b-262)){O.ub(a.window,a.window,b,b,0);a.Eb-=b;a.v-=b;a.va-=b;var e=c=a.vc;do{var f=a.head[--e];a.head[e]=f>=b?f-b:0}while(--c);e=c=b;do f=a.Ia[--e],a.Ia[e]=f>=b?f-b:0;while(--c);d+=b}if(a.M.na===0)break;e=a.M;c=a.window;f=a.v+a.D;var g=e.na;g>d&&(g=d);g===0?c=0:(e.na-=g,O.ub(c,e.input,e.nb,g,f),e.state.wrap===1?e.J=$k(e.J,c,g,f):e.state.wrap===2&&(e.J=al(e.J,c,g,f)),e.nb+=g,e.qb+=g,c=g);a.D+=c;if(a.D+a.sa>=3)for(d=a.v-a.sa,a.R=a.window[d],
a.R=(a.R<<a.Ma^a.window[d+1])&a.La;a.sa&&!(a.R=(a.R<<a.Ma^a.window[d+3-1])&a.La,a.Ia[d&a.eb]=a.head[a.R],a.head[a.R]=d,d++,a.sa--,a.D+a.sa<3););}while(a.D<262&&a.M.na!==0)}
function Ul(a,b){for(var c;;){if(a.D<262){Tl(a);if(a.D<262&&b===0)return 1;if(a.D===0)break}c=0;a.D>=3&&(a.R=(a.R<<a.Ma^a.window[a.v+3-1])&a.La,c=a.Ia[a.v&a.eb]=a.head[a.R],a.head[a.R]=a.v);c!==0&&a.v-c<=a.la-262&&(a.T=Sl(a,c));if(a.T>=3)if(c=Ml(a,a.v-a.Eb,a.T-3),a.D-=a.T,a.T<=a.Sc&&a.D>=3){a.T--;do a.v++,a.R=(a.R<<a.Ma^a.window[a.v+3-1])&a.La,a.Ia[a.v&a.eb]=a.head[a.R],a.head[a.R]=a.v;while(--a.T!==0);a.v++}else a.v+=a.T,a.T=0,a.R=a.window[a.v],a.R=(a.R<<a.Ma^a.window[a.v+1])&a.La;else c=Ml(a,0,
a.window[a.v]),a.D--,a.v++;if(c&&(Ql(a,!1),a.M.S===0))return 1}a.sa=a.v<2?a.v:2;return b===4?(Ql(a,!0),a.M.S===0?3:4):a.ya&&(Ql(a,!1),a.M.S===0)?1:2}
function Vl(a,b){for(var c,d;;){if(a.D<262){Tl(a);if(a.D<262&&b===0)return 1;if(a.D===0)break}c=0;a.D>=3&&(a.R=(a.R<<a.Ma^a.window[a.v+3-1])&a.La,c=a.Ia[a.v&a.eb]=a.head[a.R],a.head[a.R]=a.v);a.wa=a.T;a.Fd=a.Eb;a.T=2;c!==0&&a.wa<a.Sc&&a.v-c<=a.la-262&&(a.T=Sl(a,c),a.T<=5&&(a.strategy===1||a.T===3&&a.v-a.Eb>4096)&&(a.T=2));if(a.wa>=3&&a.T<=a.wa){d=a.v+a.D-3;c=Ml(a,a.v-1-a.Fd,a.wa-3);a.D-=a.wa-1;a.wa-=2;do++a.v<=d&&(a.R=(a.R<<a.Ma^a.window[a.v+3-1])&a.La,a.Ia[a.v&a.eb]=a.head[a.R],a.head[a.R]=a.v);
while(--a.wa!==0);a.lb=0;a.T=2;a.v++;if(c&&(Ql(a,!1),a.M.S===0))return 1}else if(a.lb){if((c=Ml(a,0,a.window[a.v-1]))&&Ql(a,!1),a.v++,a.D--,a.M.S===0)return 1}else a.lb=1,a.v++,a.D--}a.lb&&(Ml(a,0,a.window[a.v-1]),a.lb=0);a.sa=a.v<2?a.v:2;return b===4?(Ql(a,!0),a.M.S===0?3:4):a.ya&&(Ql(a,!1),a.M.S===0)?1:2}
function Wl(a,b){for(var c,d,e,f=a.window;;){if(a.D<=258){Tl(a);if(a.D<=258&&b===0)return 1;if(a.D===0)break}a.T=0;if(a.D>=3&&a.v>0&&(d=a.v-1,c=f[d],c===f[++d]&&c===f[++d]&&c===f[++d])){for(e=a.v+258;c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&d<e;);a.T=258-(e-d);a.T>a.D&&(a.T=a.D)}a.T>=3?(c=Ml(a,1,a.T-3),a.D-=a.T,a.v+=a.T,a.T=0):(c=Ml(a,0,a.window[a.v]),a.D--,a.v++);if(c&&(Ql(a,!1),a.M.S===0))return 1}a.sa=0;return b===4?(Ql(a,!0),a.M.S===0?3:4):
a.ya&&(Ql(a,!1),a.M.S===0)?1:2}
function Xl(a,b){for(var c;;){if(a.D===0&&(Tl(a),a.D===0)){if(b===0)return 1;break}a.T=0;c=Ml(a,0,a.window[a.v]);a.D--;a.v++;if(c&&(Ql(a,!1),a.M.S===0))return 1}a.sa=0;return b===4?(Ql(a,!0),a.M.S===0?3:4):a.ya&&(Ql(a,!1),a.M.S===0)?1:2}
function Yl(a,b,c,d,e){this.ye=a;this.Ke=b;this.Ne=c;this.Je=d;this.ue=e}
var Zl;Zl=[new Yl(0,0,0,0,function(a,b){var c=65535;for(c>a.za-5&&(c=a.za-5);;){if(a.D<=1){Tl(a);if(a.D===0&&b===0)return 1;if(a.D===0)break}a.v+=a.D;a.D=0;var d=a.va+c;if(a.v===0||a.v>=d)if(a.D=a.v-d,a.v=d,Ql(a,!1),a.M.S===0)return 1;if(a.v-a.va>=a.la-262&&(Ql(a,!1),a.M.S===0))return 1}a.sa=0;if(b===4)return Ql(a,!0),a.M.S===0?3:4;a.v>a.va&&Ql(a,!1);return 1}),
new Yl(4,4,8,4,Ul),new Yl(4,5,16,8,Ul),new Yl(4,6,32,32,Ul),new Yl(4,4,16,16,Vl),new Yl(8,16,32,32,Vl),new Yl(8,16,128,128,Vl),new Yl(8,32,128,256,Vl),new Yl(32,128,258,1024,Vl),new Yl(32,258,258,4096,Vl)];
function $l(){this.M=null;this.status=0;this.aa=null;this.wrap=this.pending=this.Yb=this.za=0;this.I=null;this.Ba=0;this.method=8;this.Cb=-1;this.eb=this.md=this.la=0;this.window=null;this.Sd=0;this.head=this.Ia=null;this.Cd=this.wd=this.strategy=this.level=this.Sc=this.Ad=this.wa=this.D=this.Eb=this.v=this.lb=this.Fd=this.T=this.va=this.Ma=this.La=this.Oc=this.vc=this.R=0;this.ra=new O.Ja(1146);this.hb=new O.Ja(122);this.ja=new O.Ja(78);Ol(this.ra);Ol(this.hb);Ol(this.ja);this.qd=this.nc=this.wc=
null;this.Ka=new O.Ja(16);this.da=new O.Ja(573);Ol(this.da);this.Ab=this.Na=0;this.depth=new O.Ja(573);Ol(this.depth);this.ia=this.oa=this.sa=this.matches=this.Ib=this.Pa=this.Qb=this.ya=this.Vb=this.Qc=0}
function am(a,b){if(!a||!a.state||b>5||b<0)return a?Nl(a,-2):-2;var c=a.state;if(!a.output||!a.input&&a.na!==0||c.status===666&&b!==4)return Nl(a,a.S===0?-5:-2);c.M=a;var d=c.Cb;c.Cb=b;if(c.status===42)if(c.wrap===2)a.J=0,P(c,31),P(c,139),P(c,8),c.I?(P(c,(c.I.text?1:0)+(c.I.Xa?2:0)+(c.I.extra?4:0)+(c.I.name?8:0)+(c.I.comment?16:0)),P(c,c.I.time&255),P(c,c.I.time>>8&255),P(c,c.I.time>>16&255),P(c,c.I.time>>24&255),P(c,c.level===9?2:c.strategy>=2||c.level<2?4:0),P(c,c.I.os&255),c.I.extra&&c.I.extra.length&&
(P(c,c.I.extra.length&255),P(c,c.I.extra.length>>8&255)),c.I.Xa&&(a.J=al(a.J,c.aa,c.pending,0)),c.Ba=0,c.status=69):(P(c,0),P(c,0),P(c,0),P(c,0),P(c,0),P(c,c.level===9?2:c.strategy>=2||c.level<2?4:0),P(c,3),c.status=113);else{var e=8+(c.md-8<<4)<<8;e|=(c.strategy>=2||c.level<2?0:c.level<6?1:c.level===6?2:3)<<6;c.v!==0&&(e|=32);c.status=113;Rl(c,e+(31-e%31));c.v!==0&&(Rl(c,a.J>>>16),Rl(c,a.J&65535));a.J=1}if(c.status===69)if(c.I.extra){for(e=c.pending;c.Ba<(c.I.extra.length&65535)&&(c.pending!==c.za||
(c.I.Xa&&c.pending>e&&(a.J=al(a.J,c.aa,c.pending-e,e)),Pl(a),e=c.pending,c.pending!==c.za));)P(c,c.I.extra[c.Ba]&255),c.Ba++;c.I.Xa&&c.pending>e&&(a.J=al(a.J,c.aa,c.pending-e,e));c.Ba===c.I.extra.length&&(c.Ba=0,c.status=73)}else c.status=73;if(c.status===73)if(c.I.name){e=c.pending;do{if(c.pending===c.za&&(c.I.Xa&&c.pending>e&&(a.J=al(a.J,c.aa,c.pending-e,e)),Pl(a),e=c.pending,c.pending===c.za)){var f=1;break}f=c.Ba<c.I.name.length?c.I.name.charCodeAt(c.Ba++)&255:0;P(c,f)}while(f!==0);c.I.Xa&&c.pending>
e&&(a.J=al(a.J,c.aa,c.pending-e,e));f===0&&(c.Ba=0,c.status=91)}else c.status=91;if(c.status===91)if(c.I.comment){e=c.pending;do{if(c.pending===c.za&&(c.I.Xa&&c.pending>e&&(a.J=al(a.J,c.aa,c.pending-e,e)),Pl(a),e=c.pending,c.pending===c.za)){f=1;break}f=c.Ba<c.I.comment.length?c.I.comment.charCodeAt(c.Ba++)&255:0;P(c,f)}while(f!==0);c.I.Xa&&c.pending>e&&(a.J=al(a.J,c.aa,c.pending-e,e));f===0&&(c.status=103)}else c.status=103;c.status===103&&(c.I.Xa?(c.pending+2>c.za&&Pl(a),c.pending+2<=c.za&&(P(c,
a.J&255),P(c,a.J>>8&255),a.J=0,c.status=113)):c.status=113);if(c.pending!==0){if(Pl(a),a.S===0)return c.Cb=-1,0}else if(a.na===0&&(b<<1)-(b>4?9:0)<=(d<<1)-(d>4?9:0)&&b!==4)return Nl(a,-5);if(c.status===666&&a.na!==0)return Nl(a,-5);if(a.na!==0||c.D!==0||b!==0&&c.status!==666){d=c.strategy===2?Xl(c,b):c.strategy===3?Wl(c,b):Zl[c.level].ue(c,b);if(d===3||d===4)c.status=666;if(d===1||d===3)return a.S===0&&(c.Cb=-1),0;if(d===2&&(b===1?(xl(c,2,3),yl(c,256,ll),c.ia===16?(wl(c,c.oa),c.oa=0,c.ia=0):c.ia>=
8&&(c.aa[c.pending++]=c.oa&255,c.oa>>=8,c.ia-=8)):b!==5&&(xl(c,0,3),Dl(c,0,0),b===3&&(Ol(c.head),c.D===0&&(c.v=0,c.va=0,c.sa=0))),Pl(a),a.S===0))return c.Cb=-1,0}if(b!==4)return 0;if(c.wrap<=0)return 1;c.wrap===2?(P(c,a.J&255),P(c,a.J>>8&255),P(c,a.J>>16&255),P(c,a.J>>24&255),P(c,a.qb&255),P(c,a.qb>>8&255),P(c,a.qb>>16&255),P(c,a.qb>>24&255)):(Rl(c,a.J>>>16),Rl(c,a.J&65535));Pl(a);c.wrap>0&&(c.wrap=-c.wrap);return c.pending!==0?0:1}
;var bm={};bm=function(){this.input=null;this.qb=this.na=this.nb=0;this.output=null;this.jd=this.S=this.Gb=0;this.msg="";this.state=null;this.Lc=2;this.J=0};var cm=Object.prototype.toString;
function dm(a){if(!(this instanceof dm))return new dm(a);a=this.options=O.assign({level:-1,method:8,chunkSize:16384,windowBits:15,memLevel:8,strategy:0,to:""},a||{});a.raw&&a.windowBits>0?a.windowBits=-a.windowBits:a.gzip&&a.windowBits>0&&a.windowBits<16&&(a.windowBits+=16);this.err=0;this.msg="";this.ended=!1;this.chunks=[];this.M=new bm;this.M.S=0;var b=this.M;var c=a.level,d=a.method,e=a.windowBits,f=a.memLevel,g=a.strategy;if(b){var h=1;c===-1&&(c=6);e<0?(h=0,e=-e):e>15&&(h=2,e-=16);if(f<1||f>
9||d!==8||e<8||e>15||c<0||c>9||g<0||g>4)b=Nl(b,-2);else{e===8&&(e=9);var k=new $l;b.state=k;k.M=b;k.wrap=h;k.I=null;k.md=e;k.la=1<<k.md;k.eb=k.la-1;k.Oc=f+7;k.vc=1<<k.Oc;k.La=k.vc-1;k.Ma=~~((k.Oc+3-1)/3);k.window=new O.rb(k.la*2);k.head=new O.Ja(k.vc);k.Ia=new O.Ja(k.la);k.Vb=1<<f+6;k.za=k.Vb*4;k.aa=new O.rb(k.za);k.Qb=1*k.Vb;k.Qc=3*k.Vb;k.level=c;k.strategy=g;k.method=d;if(b&&b.state){b.qb=b.jd=0;b.Lc=2;c=b.state;c.pending=0;c.Yb=0;c.wrap<0&&(c.wrap=-c.wrap);c.status=c.wrap?42:113;b.J=c.wrap===2?
0:1;c.Cb=0;if(!Ll){d=Array(16);for(f=g=0;f<28;f++)for(pl[f]=g,e=0;e<1<<hl[f];e++)ol[g++]=f;ol[g-1]=f;for(f=g=0;f<16;f++)for(ql[f]=g,e=0;e<1<<il[f];e++)nl[g++]=f;for(g>>=7;f<30;f++)for(ql[f]=g<<7,e=0;e<1<<il[f]-7;e++)nl[256+g++]=f;for(e=0;e<=15;e++)d[e]=0;for(e=0;e<=143;)ll[e*2+1]=8,e++,d[8]++;for(;e<=255;)ll[e*2+1]=9,e++,d[9]++;for(;e<=279;)ll[e*2+1]=7,e++,d[7]++;for(;e<=287;)ll[e*2+1]=8,e++,d[8]++;Al(ll,287,d);for(e=0;e<30;e++)ml[e*2+1]=5,ml[e*2]=zl(e,5);sl=new rl(ll,hl,257,286,15);tl=new rl(ml,
il,0,30,15);ul=new rl([],jl,0,19,7);Ll=!0}c.wc=new vl(c.ra,sl);c.nc=new vl(c.hb,tl);c.qd=new vl(c.ja,ul);c.oa=0;c.ia=0;Bl(c);c=0}else c=Nl(b,-2);c===0&&(b=b.state,b.Sd=2*b.la,Ol(b.head),b.Sc=Zl[b.level].Ke,b.wd=Zl[b.level].ye,b.Cd=Zl[b.level].Ne,b.Ad=Zl[b.level].Je,b.v=0,b.va=0,b.D=0,b.sa=0,b.T=b.wa=2,b.lb=0,b.R=0);b=c}}else b=-2;if(b!==0)throw Error(fl[b]);a.header&&(b=this.M)&&b.state&&b.state.wrap===2&&(b.state.I=a.header);if(a.dictionary){var l;typeof a.dictionary==="string"?l=Zk(a.dictionary):
cm.call(a.dictionary)==="[object ArrayBuffer]"?l=new Uint8Array(a.dictionary):l=a.dictionary;a=this.M;f=l;g=f.length;if(a&&a.state)if(l=a.state,b=l.wrap,b===2||b===1&&l.status!==42||l.D)b=-2;else{b===1&&(a.J=$k(a.J,f,g,0));l.wrap=0;g>=l.la&&(b===0&&(Ol(l.head),l.v=0,l.va=0,l.sa=0),c=new O.rb(l.la),O.ub(c,f,g-l.la,l.la,0),f=c,g=l.la);c=a.na;d=a.nb;e=a.input;a.na=g;a.nb=0;a.input=f;for(Tl(l);l.D>=3;){f=l.v;g=l.D-2;do l.R=(l.R<<l.Ma^l.window[f+3-1])&l.La,l.Ia[f&l.eb]=l.head[l.R],l.head[l.R]=f,f++;while(--g);
l.v=f;l.D=2;Tl(l)}l.v+=l.D;l.va=l.v;l.sa=l.D;l.D=0;l.T=l.wa=2;l.lb=0;a.nb=d;a.input=e;a.na=c;l.wrap=b;b=0}else b=-2;if(b!==0)throw Error(fl[b]);this.sh=!0}}
dm.prototype.push=function(a,b){var c=this.M,d=this.options.chunkSize;if(this.ended)return!1;var e=b===~~b?b:b===!0?4:0;typeof a==="string"?c.input=Zk(a):cm.call(a)==="[object ArrayBuffer]"?c.input=new Uint8Array(a):c.input=a;c.nb=0;c.na=c.input.length;do{c.S===0&&(c.output=new O.rb(d),c.Gb=0,c.S=d);a=am(c,e);if(a!==1&&a!==0)return em(this,a),this.ended=!0,!1;if(c.S===0||c.na===0&&(e===4||e===2))if(this.options.to==="string"){var f=O.dd(c.output,c.Gb);b=f;f=f.length;if(f<65537&&(b.subarray&&Yk||!b.subarray))b=
String.fromCharCode.apply(null,O.dd(b,f));else{for(var g="",h=0;h<f;h++)g+=String.fromCharCode(b[h]);b=g}this.chunks.push(b)}else b=O.dd(c.output,c.Gb),this.chunks.push(b)}while((c.na>0||c.S===0)&&a!==1);if(e===4)return(c=this.M)&&c.state?(d=c.state.status,d!==42&&d!==69&&d!==73&&d!==91&&d!==103&&d!==113&&d!==666?a=Nl(c,-2):(c.state=null,a=d===113?Nl(c,-3):0)):a=-2,em(this,a),this.ended=!0,a===0;e===2&&(em(this,0),c.S=0);return!0};
function em(a,b){b===0&&(a.result=a.options.to==="string"?a.chunks.join(""):O.ud(a.chunks));a.chunks=[];a.err=b;a.msg=a.M.msg}
function fm(a,b){b=b||{};b.gzip=!0;b=new dm(b);b.push(a,!0);if(b.err)throw b.msg||fl[b.err];return b.result}
;function gm(a){return a?(a=a.privateDoNotAccessOrElseSafeScriptWrappedValue)?Ib(a):null:null}
function hm(a){return a?(a=a.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue)?ob(a):null:null}
;function im(a){return ob(a===null?"null":a===void 0?"undefined":a)}
;function jm(a){this.name=a}
;var km=new jm("rawColdConfigGroup");var lm=new jm("rawHotConfigGroup");function mm(a){this.F=L(a)}
v(mm,M);function nm(a){this.F=L(a)}
v(nm,M);nm.prototype.setTrackingParams=function(a){if(a!=null)if(typeof a==="string")a=a?new Fd(a,Ed):Hd||(Hd=new Fd(null,Ed));else if(a.constructor!==Fd)if(zd&&a!=null&&a instanceof Uint8Array)a instanceof Uint8Array||Array.isArray(a),a=a.length?new Fd(new Uint8Array(a),Ed):Hd||(Hd=new Fd(null,Ed));else throw Error();return xf(this,1,a)};var om=new jm("continuationCommand");var pm=new jm("webCommandMetadata");var qm=new jm("signalServiceEndpoint");var rm={Sf:"EMBEDDED_PLAYER_MODE_UNKNOWN",Pf:"EMBEDDED_PLAYER_MODE_DEFAULT",Rf:"EMBEDDED_PLAYER_MODE_PFP",Qf:"EMBEDDED_PLAYER_MODE_PFL"};var sm=new jm("feedbackEndpoint");var me={Wg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_UNKNOWN",pg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_FOR_TESTING",Gg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_RESUME_TO_HOME_TTL",Og:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_START_TO_SHORTS_ANALYSIS_SLICE",eg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_DEVICE_LAYER_SLICE",Vg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_UNIFIED_LAYER_SLICE",Yg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_VISITOR_LAYER_SLICE",Ng:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SHOW_SHEET_COMMAND_HANDLER_BLOCK",
ah:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WIZ_NEXT_MIGRATED_COMPONENT",Zg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WIZ_NEXT_CHANNEL_NAME_TOOLTIP",Jg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATION_LOCK_SUPPORTED",Qg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_THEATER_MODE_ENABLED",fh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_PIN_SUGGESTION",eh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_LONG_PRESS_EDU_TOAST",dh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_AMBIENT",Rg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TIME_WATCHED_PANEL",
Lg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SEARCH_FROM_SEARCH_BAR_OVERLAY",gh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_VOICE_SEARCH_EDU_TOAST",Pg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SUGGESTED_LANGUAGE_SELECTED",hh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_TRIGGER_SHORTS_PIP",wg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IN_ZP_VOICE_CRASHY_SET",Cg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_FAST_SWIPE_SUPPRESSED",Bg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_FAST_SWIPE_ALLOWED",Eg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_PULL_TO_REFRESH_ATTEMPT",
bh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_BLOCK_KABUKI",Fg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_TALL_SCREEN",Dg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_NORMAL_SCREEN",Wf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ACCESSIBILITY_MODE_ENABLED",Vf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ACCESSIBILITY_MODE_DISABLED",Xf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_AUTOPLAY_ENABLED",Yf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_CAST_MATCH_OCCURRED",ig:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_ELIGIBLE",lg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ENDSCREEN_TRIGGERED",
Ag:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_POSTPLAY_TRIGGERED",zg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_POSTPLAY_LACT_THRESHOLD_EXCEEDED",qg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_MATCHED_ON_REMOTE_CONNECTION",sg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_SWITCHABLE_ON_REMOTE_CONNECTION",rg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_MISATTRIBUTED_ON_REMOTE_CONNECTION",vg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_TV_IS_SIGNED_IN_ON_REMOTE_CONNECTION",Tg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TV_START_TYPE_COLD_ON_REMOTE_CONNECTION",
Ug:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TV_START_TYPE_NON_COLD_ON_REMOTE_CONNECTION",yg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ON_REMOTE_CONNECTION",dg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_VALID",ag:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_INVALID",cg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_UNDEFINED",Zf:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_DEFINED",xg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_LACT_THRESHOLD_EXCEEDED",
Kg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROUND_TRIP_HANDLING_ON_REMOTE_CONNECTION",ug:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_SWITCHED_ON_REMOTE_CONNECTION_BEFORE_APP_RELOAD",tg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_SWITCHED_ON_REMOTE_CONNECTION_AFTER_APP_RELOAD",jg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_INELIGIBLE",Sg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TVHTML5_MID_ROLL_THRESHOLD_REACHED",ng:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EXP_COBALT_HTTP3_CONFIG_PENDING",
mg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EXP_COBALT_HTTP3_CONFIG_ACTIVATED",kg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_M2_ELIGIBLE",Hg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATE_DEVICE_TO_LANDSCAPE",Ig:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATE_DEVICE_TO_PORTRAIT",hg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMBEDS_FACEOFF_UI_EVENT",og:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EXP_COBALT_HTTP3_CONFIG_RECEIVED",gg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ELIGIBLE_TO_SUPPRESS_TRANSPORT_CONTROLS_BUTTONS",
Xg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_USER_HAS_THEATER_MODE_COOKIE_ENABLED",fg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_DOCUMENT_PICTURE_IN_PICTURE_SUPPORTED",Mg:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SHORTS_NON_DEFAULT_ASPECT_RATIO"};var tm=new jm("shareEndpoint"),um=new jm("shareEntityEndpoint"),wm=new jm("shareEntityServiceEndpoint"),xm=new jm("webPlayerShareEntityServiceEndpoint");var ym=new jm("playlistEditEndpoint");var zm=new jm("modifyChannelNotificationPreferenceEndpoint");var Am=new jm("undoFeedbackEndpoint");var Bm=new jm("unsubscribeEndpoint");var Cm=new jm("subscribeEndpoint");function Dm(){var a=Em;F("yt.ads.biscotti.getId_")||E("yt.ads.biscotti.getId_",a)}
function Fm(a){E("yt.ads.biscotti.lastId_",a)}
;function Gm(a,b){b.length>1?a[b[0]]=b[1]:b.length===1&&Object.assign(a,b[0])}
;var Hm=D.window,Im,Jm,Km=(Hm==null?void 0:(Im=Hm.yt)==null?void 0:Im.config_)||(Hm==null?void 0:(Jm=Hm.ytcfg)==null?void 0:Jm.data_)||{};E("yt.config_",Km);function Lm(){Gm(Km,arguments)}
function R(a,b){return a in Km?Km[a]:b}
function Mm(a){var b=Km.EXPERIMENT_FLAGS;return b?b[a]:void 0}
;var Nm=[];function Om(a){Nm.forEach(function(b){return b(a)})}
function Pm(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){Qm(b)}}:a}
function Qm(a){var b=F("yt.logging.errors.log");b?b(a,"ERROR",void 0,void 0,void 0,void 0,void 0):(b=R("ERRORS",[]),b.push([a,"ERROR",void 0,void 0,void 0,void 0,void 0]),Lm("ERRORS",b));Om(a)}
function Rm(a,b,c,d,e){var f=F("yt.logging.errors.log");f?f(a,"WARNING",b,c,d,void 0,e):(f=R("ERRORS",[]),f.push([a,"WARNING",b,c,d,void 0,e]),Lm("ERRORS",f))}
;var Sm=/^[\w.]*$/,Tm={q:!0,search_query:!0};function Um(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(f.length===1&&f[0]||f.length===2)try{var g=Vm(f[0]||""),h=Vm(f[1]||"");if(g in c){var k=c[g];Array.isArray(k)?$b(k,h):c[g]=[k,h]}else c[g]=h}catch(r){var l=r,m=f[0],n=String(Um);l.args=[{key:m,value:f[1],query:a,method:Wm===n?"unchanged":n}];Tm.hasOwnProperty(m)||Rm(l)}}return c}
var Wm=String(Um);function Xm(a){var b=[];ug(a,function(c,d){var e=encodeURIComponent(String(d));c=Array.isArray(c)?c:[c];Tb(c,function(f){f==""?b.push(e):b.push(e+"="+encodeURIComponent(String(f)))})});
return b.join("&")}
function Ym(a){a.charAt(0)==="?"&&(a=a.substring(1));return Um(a,"&")}
function Zm(a){return a.indexOf("?")!==-1?(a=(a||"").split("#")[0],a=a.split("?",2),Ym(a.length>1?a[1]:a[0])):{}}
function $m(a,b){return an(a,b||{},!0)}
function an(a,b,c){var d=a.split("#",2);a=d[0];d=d.length>1?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=Ym(e[1]||"");for(var f in b)!c&&e!==null&&f in e||(e[f]=b[f]);return rc(a,e)+d}
function bn(a){if(!b)var b=window.location.href;var c=a.match(ic)[1]||null,d=kc(a);c&&d?(a=a.match(ic),b=b.match(ic),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?kc(b)===d&&(Number(b.match(ic)[4]||null)||null)===(Number(a.match(ic)[4]||null)||null):!0;return a}
function Vm(a){return a&&a.match(Sm)?a:fc(a)}
;function cn(a){var b=dn;a=a===void 0?F("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=Sj;e.flash="0";a:{try{var f=b.h.top.location.href}catch(Ia){f=2;break a}f=f?f===b.i.location.href?0:1:2}e=(e.frm=f,e);try{e.u_tz=-(new Date).getTimezoneOffset();var g=g===void 0?Kj:g;try{var h=g.history.length}catch(Ia){h=0}e.u_his=h;var k;e.u_h=(k=Kj.screen)==null?void 0:k.height;var l;e.u_w=(l=Kj.screen)==null?void 0:l.width;var m;e.u_ah=(m=Kj.screen)==null?void 0:m.availHeight;var n;e.u_aw=
(n=Kj.screen)==null?void 0:n.availWidth;var r;e.u_cd=(r=Kj.screen)==null?void 0:r.colorDepth}catch(Ia){}var t;h=b.h;try{var w=h.screenX;var x=h.screenY}catch(Ia){}try{var z=h.outerWidth;var G=h.outerHeight}catch(Ia){}try{var H=h.innerWidth;var S=h.innerHeight}catch(Ia){}try{var Z=h.screenLeft;var mb=h.screenTop}catch(Ia){}try{H=h.innerWidth,S=h.innerHeight}catch(Ia){}try{var Rb=h.screen.availWidth;var Wa=h.screen.availTop}catch(Ia){}w=[Z,mb,w,x,Rb,Wa,z,G,H,S];try{var Cb=(b.h.top||window).document,
Xa=Cb.compatMode=="CSS1Compat"?Cb.documentElement:Cb.body;var Na=(new tg(Xa.clientWidth,Xa.clientHeight)).round()}catch(Ia){Na=new tg(-12245933,-12245933)}Cb=Na;Na={};var Ja=Ja===void 0?D:Ja;Xa=new Yj;"SVGElement"in Ja&&"createElementNS"in Ja.document&&Xa.set(0);x=Pj();x["allow-top-navigation-by-user-activation"]&&Xa.set(1);x["allow-popups-to-escape-sandbox"]&&Xa.set(2);Ja.crypto&&Ja.crypto.subtle&&Xa.set(3);"TextDecoder"in Ja&&"TextEncoder"in Ja&&Xa.set(4);Ja=Zj(Xa);Na.bc=Ja;Na.bih=Cb.height;Na.biw=
Cb.width;Na.brdim=w.join();b=b.i;b=b.prerendering?3:(t={visible:1,hidden:2,prerender:3,preview:4,unloaded:5,"":0}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""])!=null?t:0;t=(Na.vis=b,Na.wgl=!!Kj.WebGLRenderingContext,Na);c=d.call(c,e,t);c.ca_type="image";a&&(c.bid=a);return c}
var dn=new function(){var a=window.document;this.h=window;this.i=a};
E("yt.ads_.signals_.getAdSignalsString",function(a){return Xm(cn(a))});Ya();navigator.userAgent.indexOf(" (CrKey ");var en="XMLHttpRequest"in D?function(){return new XMLHttpRequest}:null;
function fn(){if(!en)return null;var a=en();return"open"in a?a:null}
function gn(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function hn(a,b){typeof a==="function"&&(a=Pm(a));return window.setTimeout(a,b)}
;var jn="client_dev_domain client_dev_expflag client_dev_regex_map client_dev_root_url client_rollout_override expflag forcedCapability jsfeat jsmode mods".split(" ");[].concat(A(jn),["client_dev_set_cookie"]);function T(a){a=kn(a);return typeof a==="string"&&a==="false"?!1:!!a}
function ln(a,b){a=kn(a);return a===void 0&&b!==void 0?b:Number(a||0)}
function kn(a){return R("EXPERIMENT_FLAGS",{})[a]}
function mn(){for(var a=[],b=R("EXPERIMENTS_FORCED_FLAGS",{}),c=y(Object.keys(b)),d=c.next();!d.done;d=c.next())d=d.value,a.push({key:d,value:String(b[d])});c=R("EXPERIMENT_FLAGS",{});d=y(Object.keys(c));for(var e=d.next();!e.done;e=d.next())e=e.value,e.startsWith("force_")&&b[e]===void 0&&a.push({key:e,value:String(c[e])});return a}
;var nn={Authorization:"AUTHORIZATION","X-Goog-EOM-Visitor-Id":"EOM_VISITOR_DATA","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Domain-Admin-State":"DOMAIN_ADMIN_STATE","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL",
"X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-Goog-AuthUser":"SESSION_INDEX","X-Goog-PageId":"DELEGATED_SESSION_ID"},on="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" ").concat(A(jn)),pn=!1;function qn(a,b,c,d,e,f,g,h){function k(){(l&&"readyState"in l?l.readyState:0)===4&&b&&Pm(b)(l)}
c=c===void 0?"GET":c;d=d===void 0?"":d;h=h===void 0?!1:h;var l=fn();if(!l)return null;"onloadend"in l?l.addEventListener("loadend",k,!1):l.onreadystatechange=k;T("debug_forward_web_query_parameters")&&(a=rn(a));l.open(c,a,!0);f&&(l.responseType=f);g&&(l.withCredentials=!0);c=c==="POST"&&(window.FormData===void 0||!(d instanceof FormData));if(e=sn(a,e))for(var m in e)l.setRequestHeader(m,e[m]),"content-type"===m.toLowerCase()&&(c=!1);c&&l.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
if(h&&"setAttributionReporting"in XMLHttpRequest.prototype){a={eventSourceEligible:!0,triggerEligible:!1};try{l.setAttributionReporting(a)}catch(n){Rm(n)}}l.send(d);return l}
function sn(a,b){b=b===void 0?{}:b;var c=bn(a),d=R("INNERTUBE_CLIENT_NAME"),e=T("web_ajax_ignore_global_headers_if_set"),f;for(f in nn){var g=R(nn[f]),h=f==="X-Goog-AuthUser"||f==="X-Goog-PageId";f!=="X-Goog-Visitor-Id"||g||(g=R("VISITOR_DATA"));var k;if(!(k=!g)){if(!(k=c||(kc(a)?!1:!0))){k=a;var l;if(l=T("add_auth_headers_to_remarketing_google_dot_com_ping")&&f==="Authorization"&&(d==="TVHTML5"||d==="TVHTML5_UNPLUGGED"||d==="TVHTML5_SIMPLY"))l=kc(k),l=l!==null?l.split(".").reverse():null,l=l===null?
!1:l[1]==="google"?!0:l[2]==="google"?l[0]==="au"&&l[1]==="com"?!0:l[0]==="uk"&&l[1]==="co"?!0:!1:!1;l&&(k=lc(k)||"",k=k.split("/"),k="/"+(k.length>1?k[1]:""),l=k==="/pagead");k=l?!0:!1}k=!k}k||e&&b[f]!==void 0||d==="TVHTML5_UNPLUGGED"&&h||(b[f]=g)}"X-Goog-EOM-Visitor-Id"in b&&"X-Goog-Visitor-Id"in b&&delete b["X-Goog-Visitor-Id"];if(c||!kc(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());if(c||!kc(a)){try{var m=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(n){}m&&
(b["X-YouTube-Time-Zone"]=m)}document.location.hostname.endsWith("youtubeeducation.com")||!c&&kc(a)||(b["X-YouTube-Ad-Signals"]=Xm(cn()));return b}
function tn(a,b){b.method="POST";b.postParams||(b.postParams={});return un(a,b)}
function un(a,b){var c=b.format||"JSON";a=vn(a,b);var d=wn(a,b),e=!1,f=xn(a,function(k){if(!e){e=!0;h&&window.clearTimeout(h);var l=gn(k),m=null,n=400<=k.status&&k.status<500,r=500<=k.status&&k.status<600;if(l||n||r)m=yn(a,c,k,b.convertToSafeHtml);l&&(l=zn(c,k,m));m=m||{};n=b.context||D;l?b.onSuccess&&b.onSuccess.call(n,k,m):b.onError&&b.onError.call(n,k,m);b.onFinish&&b.onFinish.call(n,k,m)}},b.method,d,b.headers,b.responseType,b.withCredentials);
d=b.timeout||0;if(b.onTimeout&&d>0){var g=b.onTimeout;var h=hn(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||D,f))},d)}return f}
function vn(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=R("XSRF_FIELD_NAME");if(b=b.urlParams)b[c]&&delete b[c],a=$m(a,b);return a}
function wn(a,b){var c=R("XSRF_FIELD_NAME"),d=R("XSRF_TOKEN"),e=b.postBody||"",f=b.postParams,g=R("XSRF_FIELD_NAME"),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||kc(a)&&!b.withCredentials&&kc(a)!==document.location.hostname||b.method!=="POST"||h&&h!=="application/x-www-form-urlencoded"||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);(T("ajax_parse_query_data_only_when_filled")&&f&&Object.keys(f).length>0||f)&&typeof e==="string"&&(e=Ym(e),Fg(e,f),e=b.postBodyFormat&&b.postBodyFormat===
"JSON"?JSON.stringify(e):qc(e));f=e||f&&!yg(f);!pn&&f&&b.method!=="POST"&&(pn=!0,Qm(Error("AJAX request with postData should use POST")));return e}
function yn(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,Rm(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&a.indexOf("json")>=0&&(f.substring(0,5)===")]}'\n"&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?An(a):null)e={},Tb(a.getElementsByTagName("*"),function(g){e[g.tagName]=Bn(g)})}d&&Cn(e);
return e}
function Cn(a){if(Oa(a))for(var b in a){var c;(c=b==="html_content")||(c=b.length-5,c=c>=0&&b.indexOf("_html",c)==c);if(c){c=a[b];var d=lb();c=d?d.createHTML(c):c;a[b]=new Fb(c)}else Cn(a[b])}}
function zn(a,b,c){if(b&&b.status===204)return!0;switch(a){case "JSON":return!!c;case "XML":return Number(c&&c.return_code)===0;case "RAW":return!0;default:return!!c}}
function An(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&a.length>0?a[0]:null:null}
function Bn(a){var b="";Tb(a.childNodes,function(c){b+=c.nodeValue});
return b}
function rn(a){var b=window.location.search,c=kc(a);T("debug_handle_relative_url_for_query_forward_killswitch")||!c&&bn(a)&&(c=document.location.hostname);var d=lc(a);d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=Ym(b),f={};Tb(on,function(g){e[g]&&(f[g]=e[g])});
return an(a,f||{},!1)}
var xn=qn;var Dn=[{Tc:function(a){return"Cannot read property '"+a.key+"'"},
zc:{Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}],TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,
groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]},{regexp:/(null) is not an object \(evaluating '(?:([^.]+)\.)?([^']+)'\)/,groups:["value","base","key"]}]}},{Tc:function(a){return"Cannot call '"+a.key+"'"},
zc:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}},{Tc:function(a){return a.key+" is not defined"},
zc:{ReferenceError:[{regexp:/(.*) is not defined/,groups:["key"]},{regexp:/Can't find variable: (.*)/,groups:["key"]}]}}];var Fn={Za:[],Ua:[{callback:En,weight:500}]};function En(a){if(a.name==="JavaException")return!0;a=a.stack;return a.includes("chrome://")||a.includes("chrome-extension://")||a.includes("moz-extension://")}
;function Gn(){this.Ua=[];this.Za=[]}
var Hn;function In(){if(!Hn){var a=Hn=new Gn;a.Za.length=0;a.Ua.length=0;Fn.Za&&a.Za.push.apply(a.Za,Fn.Za);Fn.Ua&&a.Ua.push.apply(a.Ua,Fn.Ua)}return Hn}
;var Jn=new N;function Kn(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=Ln(b);if(e===Infinity)break;var f=e>>3;switch(e&7){case 0:e=Ln(b);if(f===2)return e;break;case 1:if(f===2)return;d+=8;break;case 2:e=Ln(b);if(f===2)return a.substr(d,e);d+=e;break;case 5:if(f===2)return;d+=4;break;default:return}}while(d<c)}
function Ln(a){var b=a(),c=b&127;if(b<128)return c;b=a();c|=(b&127)<<7;if(b<128)return c;b=a();c|=(b&127)<<14;if(b<128)return c;b=a();return b<128?c|(b&127)<<21:Infinity}
;function Mn(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=Nn(d,a[d],b,c),e>500));d++);d=e}else if(typeof a==="object")for(e in a){if(a[e]){var f=e;var g=a[e],h=b,k=c;f=typeof g!=="string"||f!=="clickTrackingParams"&&f!=="trackingParams"?0:(g=Kn(atob(g.replace(/-/g,"+").replace(/_/g,"/"))))?Nn(f+".ve",g,h,k):0;d+=f;d+=Nn(e,a[e],b,c);if(d>500)break}}else c[b]=On(a),d+=c[b].length;else c[b]=On(a),d+=c[b].length;return d}
function Nn(a,b,c,d){c+="."+a;a=On(b);d[c]=a;return c.length+a.length}
function On(a){try{return(typeof a==="string"?a:String(JSON.stringify(a))).substr(0,500)}catch(b){return"unable to serialize "+typeof a+" ("+b.message+")"}}
;function Pn(a){var b=this;this.i=void 0;this.h=!1;a.addEventListener("beforeinstallprompt",function(c){c.preventDefault();b.i=c});
a.addEventListener("appinstalled",function(){b.h=!0},{once:!0})}
function Qn(){if(!D.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return D.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":D.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":D.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":D.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;function Rn(){this.Nd=!0}
function Sn(){Rn.instance||(Rn.instance=new Rn);return Rn.instance}
function Tn(a,b){a={};var c=[];"USER_SESSION_ID"in Km&&c.push({key:"u",value:R("USER_SESSION_ID")});if(c=mg(c))a.Authorization=c,c=b=b==null?void 0:b.sessionIndex,c===void 0&&(c=Number(R("SESSION_INDEX",0)),c=isNaN(c)?0:c),T("voice_search_auth_header_removal")||(a["X-Goog-AuthUser"]=c.toString()),"INNERTUBE_HOST_OVERRIDE"in Km||(a["X-Origin"]=window.location.origin),b===void 0&&"DELEGATED_SESSION_ID"in Km&&(a["X-Goog-PageId"]=R("DELEGATED_SESSION_ID"));return a}
;var Un={identityType:"UNAUTHENTICATED_IDENTITY_TYPE_UNKNOWN"};function Vn(a,b,c,d,e){jg.set(""+a,b,{Xb:c,path:"/",domain:d===void 0?"youtube.com":d,secure:e===void 0?!1:e})}
function Wn(a){return jg.get(""+a,void 0)}
function Xn(a,b,c){jg.remove(""+a,b===void 0?"/":b,c===void 0?"youtube.com":c)}
function Yn(){if(T("embeds_web_enable_cookie_detection_fix")){if(!D.navigator.cookieEnabled)return!1}else if(!jg.isEnabled())return!1;if(jg.h.cookie)return!0;T("embeds_web_enable_cookie_detection_fix")?jg.set("TESTCOOKIESENABLED","1",{Xb:60,cf:"none",secure:!0}):jg.set("TESTCOOKIESENABLED","1",{Xb:60});if(jg.get("TESTCOOKIESENABLED")!=="1")return!1;jg.remove("TESTCOOKIESENABLED");return!0}
;var Zn=F("ytglobal.prefsUserPrefsPrefs_")||{};E("ytglobal.prefsUserPrefsPrefs_",Zn);function $n(){this.h=R("ALT_PREF_COOKIE_NAME","PREF");this.i=R("ALT_PREF_COOKIE_DOMAIN","youtube.com");var a=Wn(this.h);a&&this.parse(a)}
var ao;function bo(){ao||(ao=new $n);return ao}
p=$n.prototype;p.get=function(a,b){co(a);eo(a);a=Zn[a]!==void 0?Zn[a].toString():null;return a!=null?a:b?b:""};
p.set=function(a,b){co(a);eo(a);if(b==null)throw Error("ExpectedNotNull");Zn[a]=b.toString()};
function fo(a){return!!((go("f"+(Math.floor(a/31)+1))||0)&1<<a%31)}
p.remove=function(a){co(a);eo(a);delete Zn[a]};
p.clear=function(){for(var a in Zn)delete Zn[a]};
function eo(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function co(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function go(a){a=Zn[a]!==void 0?Zn[a].toString():null;return a!=null&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
p.parse=function(a){a=decodeURIComponent(a).split("&");for(var b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Zn[d]=c.toString())}};var ho={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},io={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};
function jo(){var a=D.navigator;return a?a.connection:void 0}
function ko(){var a=jo();if(a){var b=ho[a.type||"unknown"]||"CONN_UNKNOWN";a=ho[a.effectiveType||"unknown"]||"CONN_UNKNOWN";b==="CONN_CELLULAR_UNKNOWN"&&a!=="CONN_UNKNOWN"&&(b=a);if(b!=="CONN_UNKNOWN")return b;if(a!=="CONN_UNKNOWN")return a}}
function lo(){var a=jo();if(a!=null&&a.effectiveType)return io.hasOwnProperty(a.effectiveType)?io[a.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN"}
;function U(a){var b=C.apply(1,arguments);var c=Error.call(this,a);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.args=[].concat(A(b));Object.setPrototypeOf(this,this.constructor.prototype)}
v(U,Error);function mo(){try{return no(),!0}catch(a){return!1}}
function no(a){if(R("DATASYNC_ID")!==void 0)return R("DATASYNC_ID");throw new U("Datasync ID not set",a===void 0?"unknown":a);}
;function oo(){}
function po(a,b){return Xj.Sa(a,0,b)}
oo.prototype.pa=function(a,b){return this.Sa(a,1,b)};
oo.prototype.Mb=function(a){var b=F("yt.scheduler.instance.addImmediateJob");b?b(a):a()};var qo=ln("web_emulated_idle_callback_delay",300),ro=1E3/60-3,so=[8,5,4,3,2,1,0];
function to(a){a=a===void 0?{}:a;I.call(this);this.i=[];this.j={};this.Z=this.h=0;this.Y=this.u=!1;this.P=[];this.U=this.ha=!1;for(var b=y(so),c=b.next();!c.done;c=b.next())this.i[c.value]=[];this.o=0;this.Ic=a.timeout||1;this.G=ro;this.A=0;this.xa=this.Pe.bind(this);this.Lb=this.wf.bind(this);this.Qa=this.Yd.bind(this);this.Ra=this.ze.bind(this);this.fb=this.We.bind(this);this.Fa=!!window.requestIdleCallback&&!!window.cancelIdleCallback&&!T("disable_scheduler_requestIdleCallback");(this.ma=a.useRaf!==
!1&&!!window.requestAnimationFrame)&&document.addEventListener("visibilitychange",this.xa)}
v(to,I);p=to.prototype;p.Mb=function(a){var b=Ya();uo(this,a);a=Ya()-b;this.u||(this.G-=a)};
p.Sa=function(a,b,c){++this.Z;if(b===10)return this.Mb(a),this.Z;var d=this.Z;this.j[d]=a;this.u&&!c?this.P.push({id:d,priority:b}):(this.i[b].push(d),this.Y||this.u||(this.h!==0&&vo(this)!==this.A&&this.stop(),this.start()));return d};
p.qa=function(a){delete this.j[a]};
function wo(a){a.P.length=0;for(var b=5;b>=0;b--)a.i[b].length=0;a.i[8].length=0;a.j={};a.stop()}
p.isHidden=function(){return!!document.hidden||!1};
function xo(a){return!a.isHidden()&&a.ma}
function vo(a){if(a.i[8].length){if(a.U)return 4;if(xo(a))return 3}for(var b=5;b>=a.o;b--)if(a.i[b].length>0)return b>0?xo(a)?3:2:1;return 0}
p.Ha=function(a){var b=F("yt.logging.errors.log");b&&b(a)};
function uo(a,b){try{b()}catch(c){a.Ha(c)}}
function yo(a){for(var b=y(so),c=b.next();!c.done;c=b.next())if(a.i[c.value].length)return!0;return!1}
p.ze=function(a){var b=void 0;a&&(b=a.timeRemaining());this.ha=!0;zo(this,b);this.ha=!1};
p.wf=function(){zo(this)};
p.Yd=function(){Ao(this)};
p.We=function(a){this.U=!0;var b=vo(this);b===4&&b!==this.A&&(this.stop(),this.start());zo(this,void 0,a);this.U=!1};
p.Pe=function(){this.isHidden()||Ao(this);this.h&&(this.stop(),this.start())};
function Ao(a){a.stop();a.u=!0;for(var b=Ya(),c=a.i[8];c.length;){var d=c.shift(),e=a.j[d];delete a.j[d];e&&uo(a,e)}Bo(a);a.u=!1;yo(a)&&a.start();b=Ya()-b;a.G-=b}
function Bo(a){for(var b=0,c=a.P.length;b<c;b++){var d=a.P[b];a.i[d.priority].push(d.id)}a.P.length=0}
function zo(a,b,c){a.U&&a.A===4&&a.h||a.stop();a.u=!0;b=Ya()+(b||a.G);for(var d=a.i[5];d.length;){var e=d.shift(),f=a.j[e];delete a.j[e];if(f){e=a;try{f(c)}catch(l){e.Ha(l)}}}for(d=a.i[4];d.length;)c=d.shift(),f=a.j[c],delete a.j[c],f&&uo(a,f);d=a.ha?0:1;d=a.o>d?a.o:d;if(!(Ya()>=b)){do{a:{c=a;f=d;for(e=3;e>=f;e--)for(var g=c.i[e];g.length;){var h=g.shift(),k=c.j[h];delete c.j[h];if(k){c=k;break a}}c=null}c&&uo(a,c)}while(c&&Ya()<b)}a.u=!1;Bo(a);a.G=ro;yo(a)&&a.start()}
p.start=function(){this.Y=!1;if(this.h===0)switch(this.A=vo(this),this.A){case 1:var a=this.Ra;this.h=this.Fa?window.requestIdleCallback(a,{timeout:3E3}):window.setTimeout(a,qo);break;case 2:this.h=window.setTimeout(this.Lb,this.Ic);break;case 3:this.h=window.requestAnimationFrame(this.fb);break;case 4:this.h=window.setTimeout(this.Qa,0)}};
p.pause=function(){this.stop();this.Y=!0};
p.stop=function(){if(this.h){switch(this.A){case 1:var a=this.h;this.Fa?window.cancelIdleCallback(a):window.clearTimeout(a);break;case 2:case 4:window.clearTimeout(this.h);break;case 3:window.cancelAnimationFrame(this.h)}this.h=0}};
p.ba=function(){wo(this);this.stop();this.ma&&document.removeEventListener("visibilitychange",this.xa);I.prototype.ba.call(this)};var Co=F("yt.scheduler.instance.timerIdMap_")||{},Do=ln("kevlar_tuner_scheduler_soft_state_timer_ms",800),Eo=0,Fo=0;function Go(){var a=F("ytglobal.schedulerInstanceInstance_");if(!a||a.ea)a=new to(R("scheduler")||{}),E("ytglobal.schedulerInstanceInstance_",a);return a}
function Ho(){Io();var a=F("ytglobal.schedulerInstanceInstance_");a&&(xc(a),E("ytglobal.schedulerInstanceInstance_",null))}
function Io(){wo(Go());for(var a in Co)Co.hasOwnProperty(a)&&delete Co[Number(a)]}
function Jo(a,b,c){if(!c)return c=c===void 0,-Go().Sa(a,b,c);var d=window.setTimeout(function(){var e=Go().Sa(a,b);Co[d]=e},c);
return d}
function Ko(a){Go().Mb(a)}
function Lo(a){var b=Go();if(a<0)b.qa(-a);else{var c=Co[a];c?(b.qa(c),delete Co[a]):window.clearTimeout(a)}}
function Mo(){No()}
function No(){window.clearTimeout(Eo);Go().start()}
function Oo(){Go().pause();window.clearTimeout(Eo);Eo=window.setTimeout(Mo,Do)}
function Po(){window.clearTimeout(Fo);Fo=window.setTimeout(function(){Qo(0)},Do)}
function Qo(a){Po();var b=Go();b.o=a;b.start()}
function Ro(a){Po();var b=Go();b.o>a&&(b.o=a,b.start())}
function So(){window.clearTimeout(Fo);var a=Go();a.o=0;a.start()}
;function To(){oo.apply(this,arguments)}
v(To,oo);function Uo(){To.instance||(To.instance=new To);return To.instance}
To.prototype.Sa=function(a,b,c){c!==void 0&&Number.isNaN(Number(c))&&(c=void 0);var d=F("yt.scheduler.instance.addJob");return d?d(a,b,c):c===void 0?(a(),NaN):hn(a,c||0)};
To.prototype.qa=function(a){if(a===void 0||!Number.isNaN(Number(a))){var b=F("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}};
To.prototype.start=function(){var a=F("yt.scheduler.instance.start");a&&a()};
To.prototype.pause=function(){var a=F("yt.scheduler.instance.pause");a&&a()};
var Xj=Uo();
T("web_scheduler_auto_init")&&!F("yt.scheduler.initialized")&&(E("yt.scheduler.instance.dispose",Ho),E("yt.scheduler.instance.addJob",Jo),E("yt.scheduler.instance.addImmediateJob",Ko),E("yt.scheduler.instance.cancelJob",Lo),E("yt.scheduler.instance.cancelAllJobs",Io),E("yt.scheduler.instance.start",No),E("yt.scheduler.instance.pause",Oo),E("yt.scheduler.instance.setPriorityThreshold",Qo),E("yt.scheduler.instance.enablePriorityThreshold",Ro),E("yt.scheduler.instance.clearPriorityThreshold",So),E("yt.scheduler.initialized",
!0));function Vo(a){var b=new yk;this.h=(a=b.isAvailable()?a?new zk(b,a):b:null)?new tk(a):null;this.i=document.domain||window.location.hostname}
Vo.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.h)try{this.h.set(a,b,Date.now()+c*1E3);return}catch(f){}var e="";if(d)try{e=escape((new Ti).serialize(b))}catch(f){return}else e=escape(b);Vn(a,e,c,this.i)};
Vo.prototype.get=function(a,b){var c=void 0,d=!this.h;if(!d)try{c=this.h.get(a)}catch(e){d=!0}if(d&&(c=Wn(a))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
Vo.prototype.remove=function(a){this.h&&this.h.remove(a);Xn(a,"/",this.i)};var Wo=function(){var a;return function(){a||(a=new Vo("ytidb"));return a}}();
function Xo(){var a;return(a=Wo())==null?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
;var Yo=[],Zo,$o=!1;function ap(){var a={};for(Zo=new bp(a.handleError===void 0?cp:a.handleError,a.logEvent===void 0?dp:a.logEvent);Yo.length>0;)switch(a=Yo.shift(),a.type){case "ERROR":Zo.Ha(a.payload);break;case "EVENT":Zo.logEvent(a.eventType,a.payload)}}
function ep(a){$o||(Zo?Zo.Ha(a):(Yo.push({type:"ERROR",payload:a}),Yo.length>10&&Yo.shift()))}
function fp(a,b){$o||(Zo?Zo.logEvent(a,b):(Yo.push({type:"EVENT",eventType:a,payload:b}),Yo.length>10&&Yo.shift()))}
;function gp(a){if(a.indexOf(":")>=0)throw Error("Database name cannot contain ':'");}
function hp(a){return a.substr(0,a.indexOf(":"))||a}
;var ip=qd||rd;function jp(a){var b=$c();return b?b.toLowerCase().indexOf(a)>=0:!1}
;var kp={},lp=(kp.AUTH_INVALID="No user identifier specified.",kp.EXPLICIT_ABORT="Transaction was explicitly aborted.",kp.IDB_NOT_SUPPORTED="IndexedDB is not supported.",kp.MISSING_INDEX="Index not created.",kp.MISSING_OBJECT_STORES="Object stores not created.",kp.DB_DELETED_BY_MISSING_OBJECT_STORES="Database is deleted because expected object stores were not created.",kp.DB_REOPENED_BY_MISSING_OBJECT_STORES="Database is reopened because expected object stores were not created.",kp.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",
kp.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",kp.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",kp.EXECUTE_TRANSACTION_ON_CLOSED_DB="Can't start a transaction on a closed database",kp.INCOMPATIBLE_DB_VERSION="The binary is incompatible with the database version",kp),mp={},np=(mp.AUTH_INVALID="ERROR",mp.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",mp.EXPLICIT_ABORT="IGNORED",mp.IDB_NOT_SUPPORTED="ERROR",mp.MISSING_INDEX=
"WARNING",mp.MISSING_OBJECT_STORES="ERROR",mp.DB_DELETED_BY_MISSING_OBJECT_STORES="WARNING",mp.DB_REOPENED_BY_MISSING_OBJECT_STORES="WARNING",mp.QUOTA_EXCEEDED="WARNING",mp.QUOTA_MAYBE_EXCEEDED="WARNING",mp.UNKNOWN_ABORT="WARNING",mp.INCOMPATIBLE_DB_VERSION="WARNING",mp),op={},pp=(op.AUTH_INVALID=!1,op.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,op.EXPLICIT_ABORT=!1,op.IDB_NOT_SUPPORTED=!1,op.MISSING_INDEX=!1,op.MISSING_OBJECT_STORES=!1,op.DB_DELETED_BY_MISSING_OBJECT_STORES=!1,op.DB_REOPENED_BY_MISSING_OBJECT_STORES=
!1,op.QUOTA_EXCEEDED=!1,op.QUOTA_MAYBE_EXCEEDED=!0,op.UNKNOWN_ABORT=!0,op.INCOMPATIBLE_DB_VERSION=!1,op);function qp(a,b,c,d,e){b=b===void 0?{}:b;c=c===void 0?lp[a]:c;d=d===void 0?np[a]:d;e=e===void 0?pp[a]:e;U.call(this,c,Object.assign({},{name:"YtIdbKnownError",isSw:self.document===void 0,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.h=e;Object.setPrototypeOf(this,qp.prototype)}
v(qp,U);function rp(a,b){qp.call(this,"MISSING_OBJECT_STORES",{expectedObjectStores:b,foundObjectStores:a},lp.MISSING_OBJECT_STORES);Object.setPrototypeOf(this,rp.prototype)}
v(rp,qp);function sp(a,b){var c=Error.call(this);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.index=a;this.objectStore=b;Object.setPrototypeOf(this,sp.prototype)}
v(sp,Error);var tp=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function up(a,b,c,d){b=hp(b);var e=a instanceof Error?a:Error("Unexpected error: "+a);if(e instanceof qp)return e;a={objectStoreNames:c,dbName:b,dbVersion:d};if(e.name==="QuotaExceededError")return new qp("QUOTA_EXCEEDED",a);if(sd&&e.name==="UnknownError")return new qp("QUOTA_MAYBE_EXCEEDED",a);if(e instanceof sp)return new qp("MISSING_INDEX",Object.assign({},a,{objectStore:e.objectStore,index:e.index}));if(e.name==="InvalidStateError"&&tp.some(function(f){return e.message.includes(f)}))return new qp("EXECUTE_TRANSACTION_ON_CLOSED_DB",
a);
if(e.name==="AbortError")return new qp("UNKNOWN_ABORT",a,e.message);e.args=[Object.assign({},a,{name:"IdbError",Ed:e.name})];e.level="WARNING";return e}
function vp(a,b,c){var d=Xo();return new qp("IDB_NOT_SUPPORTED",{context:{caller:a,publicName:b,version:c,hasSucceededOnce:d==null?void 0:d.hasSucceededOnce}})}
;function wp(a){if(!a)throw Error();throw a;}
function xp(a){return a}
function yp(a){this.h=a}
function zp(a){function b(e){if(d.state.status==="PENDING"){d.state={status:"REJECTED",reason:e};e=y(d.i);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if(d.state.status==="PENDING"){d.state={status:"FULFILLED",value:e};e=y(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.state={status:"PENDING"};this.h=[];this.i=[];a=a.h;try{a(c,b)}catch(e){b(e)}}
zp.all=function(a){return new zp(new yp(function(b,c){var d=[],e=a.length;e===0&&b(d);for(var f={Bb:0};f.Bb<a.length;f={Bb:f.Bb},++f.Bb)zp.resolve(a[f.Bb]).then(function(g){return function(h){d[g.Bb]=h;e--;e===0&&b(d)}}(f)).catch(function(g){c(g)})}))};
zp.resolve=function(a){return new zp(new yp(function(b,c){a instanceof zp?a.then(b,c):b(a)}))};
zp.reject=function(a){return new zp(new yp(function(b,c){c(a)}))};
zp.prototype.then=function(a,b){var c=this,d=a!=null?a:xp,e=b!=null?b:wp;return new zp(new yp(function(f,g){c.state.status==="PENDING"?(c.h.push(function(){Ap(c,c,d,f,g)}),c.i.push(function(){Bp(c,c,e,f,g)})):c.state.status==="FULFILLED"?Ap(c,c,d,f,g):c.state.status==="REJECTED"&&Bp(c,c,e,f,g)}))};
zp.prototype.catch=function(a){return this.then(void 0,a)};
function Ap(a,b,c,d,e){try{if(a.state.status!=="FULFILLED")throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof zp?Cp(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Bp(a,b,c,d,e){try{if(a.state.status!=="REJECTED")throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof zp?Cp(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Cp(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof zp?Cp(a,b,f,d,e):d(f)},function(f){e(f)})}
;function Dp(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function Ep(a){return new Promise(function(b,c){Dp(a,b,c)})}
function Fp(a){return new zp(new yp(function(b,c){Dp(a,b,c)}))}
;function Gp(a,b){return new zp(new yp(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()}))}
;var Hp=window,V=Hp.ytcsi&&Hp.ytcsi.now?Hp.ytcsi.now:Hp.performance&&Hp.performance.timing&&Hp.performance.now&&Hp.performance.timing.navigationStart?function(){return Hp.performance.timing.navigationStart+Hp.performance.now()}:function(){return(new Date).getTime()};function Ip(a,b){this.h=a;this.options=b;this.transactionCount=0;this.j=Math.round(V());this.i=!1}
p=Ip.prototype;p.add=function(a,b,c){return Jp(this,[a],{mode:"readwrite",ka:!0},function(d){return d.objectStore(a).add(b,c)})};
p.clear=function(a){return Jp(this,[a],{mode:"readwrite",ka:!0},function(b){return b.objectStore(a).clear()})};
p.close=function(){this.h.close();var a;((a=this.options)==null?0:a.closed)&&this.options.closed()};
p.count=function(a,b){return Jp(this,[a],{mode:"readonly",ka:!0},function(c){return c.objectStore(a).count(b)})};
function Kp(a,b,c){a=a.h.createObjectStore(b,c);return new Lp(a)}
p.delete=function(a,b){return Jp(this,[a],{mode:"readwrite",ka:!0},function(c){return c.objectStore(a).delete(b)})};
p.get=function(a,b){return Jp(this,[a],{mode:"readonly",ka:!0},function(c){return c.objectStore(a).get(b)})};
function Mp(a,b,c){return Jp(a,[b],{mode:"readwrite",ka:!0},function(d){d=d.objectStore(b);return Fp(d.h.put(c,void 0))})}
p.objectStoreNames=function(){return Array.from(this.h.objectStoreNames)};
function Jp(a,b,c,d){var e,f,g,h,k,l,m,n,r,t,w,x;return B(function(z){switch(z.h){case 1:var G={mode:"readonly",ka:!1,tag:"IDB_TRANSACTION_TAG_UNKNOWN"};typeof c==="string"?G.mode=c:Object.assign(G,c);e=G;a.transactionCount++;f=e.ka?3:1;g=0;case 2:if(h){z.B(4);break}g++;k=Math.round(V());va(z,5);l=a.h.transaction(b,e.mode);G=z.yield;var H=new Np(l);H=Op(H,d);return G.call(z,H,7);case 7:return m=z.i,n=Math.round(V()),Pp(a,k,n,g,void 0,b.join(),e),z.return(m);case 5:r=xa(z);t=Math.round(V());w=up(r,
a.h.name,b.join(),a.h.version);if((x=w instanceof qp&&!w.h)||g>=f)Pp(a,k,t,g,w,b.join(),e),h=w;z.B(2);break;case 4:return z.return(Promise.reject(h))}})}
function Pp(a,b,c,d,e,f,g){b=c-b;e?(e instanceof qp&&(e.type==="QUOTA_EXCEEDED"||e.type==="QUOTA_MAYBE_EXCEEDED")&&fp("QUOTA_EXCEEDED",{dbName:hp(a.h.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof qp&&e.type==="UNKNOWN_ABORT"&&(c-=a.j,c<0&&c>=2147483648&&(c=0),fp("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c}),a.i=!0),Qp(a,!1,d,f,b,g.tag),ep(e)):Qp(a,!0,d,f,b,g.tag)}
function Qp(a,b,c,d,e,f){fp("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.i,duration:e,isSuccessful:b,tryCount:c,tag:f===void 0?"IDB_TRANSACTION_TAG_UNKNOWN":f})}
p.getName=function(){return this.h.name};
function Lp(a){this.h=a}
p=Lp.prototype;p.add=function(a,b){return Fp(this.h.add(a,b))};
p.autoIncrement=function(){return this.h.autoIncrement};
p.clear=function(){return Fp(this.h.clear()).then(function(){})};
function Rp(a,b,c){a.h.createIndex(b,c,{unique:!1})}
p.count=function(a){return Fp(this.h.count(a))};
function Sp(a,b){return Tp(a,{query:b},function(c){return c.delete().then(function(){return Up(c)})}).then(function(){})}
p.delete=function(a){return a instanceof IDBKeyRange?Sp(this,a):Fp(this.h.delete(a))};
p.get=function(a){return Fp(this.h.get(a))};
p.index=function(a){try{return new Vp(this.h.index(a))}catch(b){if(b instanceof Error&&b.name==="NotFoundError")throw new sp(a,this.h.name);throw b;}};
p.getName=function(){return this.h.name};
p.keyPath=function(){return this.h.keyPath};
function Tp(a,b,c){a=a.h.openCursor(b.query,b.direction);return Wp(a).then(function(d){return Gp(d,c)})}
function Np(a){var b=this;this.h=a;this.i=new Map;this.aborted=!1;this.done=new Promise(function(c,d){b.h.addEventListener("complete",function(){c()});
b.h.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.h.error)});
b.h.addEventListener("abort",function(){var e=b.h.error;if(e)d(e);else if(!b.aborted){e=qp;for(var f=b.h.objectStoreNames,g=[],h=0;h<f.length;h++){var k=f.item(h);if(k===null)throw Error("Invariant: item in DOMStringList is null");g.push(k)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.h.db.name,mode:b.h.mode});d(e)}})})}
function Op(a,b){var c=new Promise(function(d,e){try{b(a).then(function(f){d(f)}).catch(e)}catch(f){e(f),a.abort()}});
return Promise.all([c,a.done]).then(function(d){return y(d).next().value})}
Np.prototype.abort=function(){this.h.abort();this.aborted=!0;throw new qp("EXPLICIT_ABORT");};
Np.prototype.objectStore=function(a){a=this.h.objectStore(a);var b=this.i.get(a);b||(b=new Lp(a),this.i.set(a,b));return b};
function Vp(a){this.h=a}
p=Vp.prototype;p.count=function(a){return Fp(this.h.count(a))};
p.delete=function(a){return Xp(this,{query:a},function(b){return b.delete().then(function(){return Up(b)})})};
p.get=function(a){return Fp(this.h.get(a))};
p.keyPath=function(){return this.h.keyPath};
p.unique=function(){return this.h.unique};
function Xp(a,b,c){a=a.h.openCursor(b.query===void 0?null:b.query,b.direction===void 0?"next":b.direction);return Wp(a).then(function(d){return Gp(d,c)})}
function Yp(a,b){this.request=a;this.cursor=b}
function Wp(a){return Fp(a).then(function(b){return b?new Yp(a,b):null})}
function Up(a){a.cursor.continue(void 0);return Wp(a.request)}
Yp.prototype.delete=function(){return Fp(this.cursor.delete()).then(function(){})};
Yp.prototype.getValue=function(){return this.cursor.value};
Yp.prototype.update=function(a){return Fp(this.cursor.update(a))};function Zp(a,b,c){return new Promise(function(d,e){function f(){r||(r=new Ip(g.result,{closed:n}));return r}
var g=b!==void 0?self.indexedDB.open(a,b):self.indexedDB.open(a);var h=c.be,k=c.blocking,l=c.tf,m=c.upgrade,n=c.closed,r;g.addEventListener("upgradeneeded",function(t){try{if(t.newVersion===null)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(g.transaction===null)throw Error("Invariant: transaction on IDbOpenDbRequest is null");t.dataLoss&&t.dataLoss!=="none"&&fp("IDB_DATA_CORRUPTED",{reason:t.dataLossMessage||"unknown reason",dbName:hp(a)});var w=f(),x=new Np(g.transaction);
m&&m(w,function(z){return t.oldVersion<z&&t.newVersion>=z},x);
x.done.catch(function(z){e(z)})}catch(z){e(z)}});
g.addEventListener("success",function(){var t=g.result;k&&t.addEventListener("versionchange",function(){k(f())});
t.addEventListener("close",function(){fp("IDB_UNEXPECTEDLY_CLOSED",{dbName:hp(a),dbVersion:t.version});l&&l()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function $p(a,b,c){c=c===void 0?{}:c;return Zp(a,b,c)}
function aq(a,b){b=b===void 0?{}:b;var c,d,e,f;return B(function(g){if(g.h==1)return va(g,2),c=self.indexedDB.deleteDatabase(a),d=b,(e=d.be)&&c.addEventListener("blocked",function(){e()}),g.yield(Ep(c),4);
if(g.h!=2)return wa(g,0);f=xa(g);throw up(f,a,"",-1);})}
;function bq(a,b){this.name=a;this.options=b;this.j=!0;this.u=this.o=0}
bq.prototype.i=function(a,b,c){c=c===void 0?{}:c;return $p(a,b,c)};
bq.prototype.delete=function(a){a=a===void 0?{}:a;return aq(this.name,a)};
function cq(a,b){return new qp("INCOMPATIBLE_DB_VERSION",{dbName:a.name,oldVersion:a.options.version,newVersion:b})}
function dq(a,b){if(!b)throw vp("openWithToken",hp(a.name));return a.open()}
bq.prototype.open=function(){function a(){var f,g,h,k,l,m,n,r,t,w;return B(function(x){switch(x.h){case 1:return g=(f=Error().stack)!=null?f:"",va(x,2),x.yield(c.i(c.name,c.options.version,e),4);case 4:for(var z=h=x.i,G=c.options,H=[],S=y(Object.keys(G.Hb)),Z=S.next();!Z.done;Z=S.next()){Z=Z.value;var mb=G.Hb[Z],Rb=mb.Xe===void 0?Number.MAX_VALUE:mb.Xe;!(z.h.version>=mb.Ob)||z.h.version>=Rb||z.h.objectStoreNames.contains(Z)||H.push(Z)}k=H;if(k.length===0){x.B(5);break}l=Object.keys(c.options.Hb);
m=h.objectStoreNames();if(c.u<ln("ytidb_reopen_db_retries",0))return c.u++,h.close(),ep(new qp("DB_REOPENED_BY_MISSING_OBJECT_STORES",{dbName:c.name,expectedObjectStores:l,foundObjectStores:m})),x.return(a());if(!(c.o<ln("ytidb_remake_db_retries",1))){x.B(6);break}c.o++;return x.yield(c.delete(),7);case 7:return ep(new qp("DB_DELETED_BY_MISSING_OBJECT_STORES",{dbName:c.name,expectedObjectStores:l,foundObjectStores:m})),x.return(a());case 6:throw new rp(m,l);case 5:return x.return(h);case 2:n=xa(x);
if(n instanceof DOMException?n.name!=="VersionError":"DOMError"in self&&n instanceof DOMError?n.name!=="VersionError":!(n instanceof Object&&"message"in n)||n.message!=="An attempt was made to open a database using a lower version than the existing version."){x.B(8);break}return x.yield(c.i(c.name,void 0,Object.assign({},e,{upgrade:void 0})),9);case 9:r=x.i;t=r.h.version;if(c.options.version!==void 0&&t>c.options.version+1)throw r.close(),c.j=!1,cq(c,t);return x.return(r);case 8:throw b(),n instanceof
Error&&!T("ytidb_async_stack_killswitch")&&(n.stack=n.stack+"\n"+g.substring(g.indexOf("\n")+1)),up(n,c.name,"",(w=c.options.version)!=null?w:-1);}})}
function b(){c.h===d&&(c.h=void 0)}
var c=this;if(!this.j)throw cq(this);if(this.h)return this.h;var d,e={blocking:function(f){f.close()},
closed:b,tf:b,upgrade:this.options.upgrade};return this.h=d=a()};var eq=new bq("YtIdbMeta",{Hb:{databases:{Ob:1}},upgrade:function(a,b){b(1)&&Kp(a,"databases",{keyPath:"actualName"})}});
function fq(a,b){var c;return B(function(d){if(d.h==1)return d.yield(dq(eq,b),2);c=d.i;return d.return(Jp(c,["databases"],{ka:!0,mode:"readwrite"},function(e){var f=e.objectStore("databases");return f.get(a.actualName).then(function(g){if(g?a.actualName!==g.actualName||a.publicName!==g.publicName||a.userIdentifier!==g.userIdentifier:1)return Fp(f.h.put(a,void 0)).then(function(){})})}))})}
function gq(a,b){var c;return B(function(d){if(d.h==1)return a?d.yield(dq(eq,b),2):d.return();c=d.i;return d.return(c.delete("databases",a))})}
function hq(a,b){var c,d;return B(function(e){return e.h==1?(c=[],e.yield(dq(eq,b),2)):e.h!=3?(d=e.i,e.yield(Jp(d,["databases"],{ka:!0,mode:"readonly"},function(f){c.length=0;return Tp(f.objectStore("databases"),{},function(g){a(g.getValue())&&c.push(g.getValue());return Up(g)})}),3)):e.return(c)})}
function iq(a){return hq(function(b){return b.publicName==="LogsDatabaseV2"&&b.userIdentifier!==void 0},a)}
function jq(a,b,c){return hq(function(d){return c?d.userIdentifier!==void 0&&!a.includes(d.userIdentifier)&&c.includes(d.publicName):d.userIdentifier!==void 0&&!a.includes(d.userIdentifier)},b)}
function kq(a){var b,c;return B(function(d){if(d.h==1)return b=no("YtIdbMeta hasAnyMeta other"),d.yield(hq(function(e){return e.userIdentifier!==void 0&&e.userIdentifier!==b},a),2);
c=d.i;return d.return(c.length>0)})}
;var lq,mq=new function(){}(new function(){});
function nq(){var a,b,c,d;return B(function(e){switch(e.h){case 1:a=Xo();if((b=a)==null?0:b.hasSucceededOnce)return e.return(!0);var f;if(f=ip)f=/WebKit\/([0-9]+)/.exec($c()),f=!!(f&&parseInt(f[1],10)>=600);f&&(f=/WebKit\/([0-9]+)/.exec($c()),f=!(f&&parseInt(f[1],10)>=602));if(f||md)return e.return(!1);try{if(c=self,!(c.indexedDB&&c.IDBIndex&&c.IDBKeyRange&&c.IDBObjectStore))return e.return(!1)}catch(g){return e.return(!1)}if(!("IDBTransaction"in self&&"objectStoreNames"in IDBTransaction.prototype))return e.return(!1);
va(e,2);d={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return e.yield(fq(d,mq),4);case 4:return e.yield(gq("yt-idb-test-do-not-use",mq),5);case 5:return e.return(!0);case 2:return xa(e),e.return(!1)}})}
function oq(){if(lq!==void 0)return lq;$o=!0;return lq=nq().then(function(a){$o=!1;var b;if((b=Wo())!=null&&b.h){var c;b={hasSucceededOnce:((c=Xo())==null?void 0:c.hasSucceededOnce)||a};var d;(d=Wo())==null||d.set("LAST_RESULT_ENTRY_KEY",b,2592E3,!0)}return a})}
function pq(){return F("ytglobal.idbToken_")||void 0}
function qq(){var a=pq();return a?Promise.resolve(a):oq().then(function(b){(b=b?mq:void 0)&&E("ytglobal.idbToken_",b);return b})}
;var rq=0;function sq(a,b){rq||(rq=Xj.pa(function(){var c,d,e,f,g;return B(function(h){switch(h.h){case 1:return h.yield(qq(),2);case 2:c=h.i;if(!c)return h.return();d=!0;va(h,3);return h.yield(jq(a,c,b),5);case 5:e=h.i;if(!e.length){d=!1;h.B(6);break}f=e[0];return h.yield(aq(f.actualName),7);case 7:return h.yield(gq(f.actualName,c),6);case 6:wa(h,4);break;case 3:g=xa(h),ep(g),d=!1;case 4:Xj.qa(rq),rq=0,d&&sq(a,b),h.h=0}})}))}
function tq(){var a;return B(function(b){return b.h==1?b.yield(qq(),2):(a=b.i)?b.return(kq(a)):b.return(!1)})}
new zj;function uq(a){if(!mo())throw a=new qp("AUTH_INVALID",{dbName:a}),ep(a),a;var b=no();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function vq(a,b,c,d){var e,f,g,h,k,l;return B(function(m){switch(m.h){case 1:return f=(e=Error().stack)!=null?e:"",m.yield(qq(),2);case 2:g=m.i;if(!g)throw h=vp("openDbImpl",a,b),T("ytidb_async_stack_killswitch")||(h.stack=h.stack+"\n"+f.substring(f.indexOf("\n")+1)),ep(h),h;gp(a);k=c?{actualName:a,publicName:a,userIdentifier:void 0}:uq(a);va(m,3);return m.yield(fq(k,g),5);case 5:return m.yield($p(k.actualName,b,d),6);case 6:return m.return(m.i);case 3:return l=xa(m),va(m,7),m.yield(gq(k.actualName,
g),9);case 9:wa(m,8);break;case 7:xa(m);case 8:throw l;}})}
function wq(a,b,c){c=c===void 0?{}:c;return vq(a,b,!1,c)}
function xq(a,b,c){c=c===void 0?{}:c;return vq(a,b,!0,c)}
function yq(a,b){b=b===void 0?{}:b;var c,d;return B(function(e){if(e.h==1)return e.yield(qq(),2);if(e.h!=3){c=e.i;if(!c)return e.return();gp(a);d=uq(a);return e.yield(aq(d.actualName,b),3)}return e.yield(gq(d.actualName,c),0)})}
function zq(a,b,c){a=a.map(function(d){return B(function(e){return e.h==1?e.yield(aq(d.actualName,b),2):e.yield(gq(d.actualName,c),0)})});
return Promise.all(a).then(function(){})}
function Aq(){var a=a===void 0?{}:a;var b,c;return B(function(d){if(d.h==1)return d.yield(qq(),2);if(d.h!=3){b=d.i;if(!b)return d.return();gp("LogsDatabaseV2");return d.yield(iq(b),3)}c=d.i;return d.yield(zq(c,a,b),0)})}
function Bq(a,b){b=b===void 0?{}:b;var c;return B(function(d){if(d.h==1)return d.yield(qq(),2);if(d.h!=3){c=d.i;if(!c)return d.return();gp(a);return d.yield(aq(a,b),3)}return d.yield(gq(a,c),0)})}
;function Cq(a,b){bq.call(this,a,b);this.options=b;gp(a)}
v(Cq,bq);function Dq(a,b){var c;return function(){c||(c=new Cq(a,b));return c}}
Cq.prototype.i=function(a,b,c){c=c===void 0?{}:c;return(this.options.shared?xq:wq)(a,b,Object.assign({},c))};
Cq.prototype.delete=function(a){a=a===void 0?{}:a;return(this.options.shared?Bq:yq)(this.name,a)};
function Eq(a,b){return Dq(a,b)}
;var Fq={},Gq=Eq("ytGcfConfig",{Hb:(Fq.coldConfigStore={Ob:1},Fq.hotConfigStore={Ob:1},Fq),shared:!1,upgrade:function(a,b){b(1)&&(Rp(Kp(a,"hotConfigStore",{keyPath:"key",autoIncrement:!0}),"hotTimestampIndex","timestamp"),Rp(Kp(a,"coldConfigStore",{keyPath:"key",autoIncrement:!0}),"coldTimestampIndex","timestamp"))},
version:1});function Hq(a){return dq(Gq(),a)}
function Iq(a,b,c){var d,e,f;return B(function(g){switch(g.h){case 1:return d={config:a,hashData:b,timestamp:V()},g.yield(Hq(c),2);case 2:return e=g.i,g.yield(e.clear("hotConfigStore"),3);case 3:return g.yield(Mp(e,"hotConfigStore",d),4);case 4:return f=g.i,g.return(f)}})}
function Jq(a,b,c,d){var e,f,g;return B(function(h){switch(h.h){case 1:return e={config:a,hashData:b,configData:c,timestamp:V()},h.yield(Hq(d),2);case 2:return f=h.i,h.yield(f.clear("coldConfigStore"),3);case 3:return h.yield(Mp(f,"coldConfigStore",e),4);case 4:return g=h.i,h.return(g)}})}
function Kq(a){var b,c;return B(function(d){return d.h==1?d.yield(Hq(a),2):d.h!=3?(b=d.i,c=void 0,d.yield(Jp(b,["coldConfigStore"],{mode:"readwrite",ka:!0},function(e){return Xp(e.objectStore("coldConfigStore").index("coldTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
function Lq(a){var b,c;return B(function(d){return d.h==1?d.yield(Hq(a),2):d.h!=3?(b=d.i,c=void 0,d.yield(Jp(b,["hotConfigStore"],{mode:"readwrite",ka:!0},function(e){return Xp(e.objectStore("hotConfigStore").index("hotTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
;function Mq(){I.call(this);this.i=[];this.h=[];var a=F("yt.gcf.config.hotUpdateCallbacks");a?(this.i=[].concat(A(a)),this.h=a):(this.h=[],E("yt.gcf.config.hotUpdateCallbacks",this.h))}
v(Mq,I);Mq.prototype.ba=function(){for(var a=y(this.i),b=a.next();!b.done;b=a.next()){var c=this.h;b=c.indexOf(b.value);b>=0&&c.splice(b,1)}this.i.length=0;I.prototype.ba.call(this)};function Nq(){this.h=0;this.i=new Mq}
function Oq(){var a;return(a=F("yt.gcf.config.hotConfigGroup"))!=null?a:R("RAW_HOT_CONFIG_GROUP")}
function Pq(a,b,c){var d,e,f;return B(function(g){switch(g.h){case 1:if(!T("start_client_gcf")){g.B(0);break}c&&(a.j=c,E("yt.gcf.config.hotConfigGroup",a.j||null));a.o(b);d=pq();if(!d){g.B(3);break}if(c){g.B(4);break}return g.yield(Lq(d),5);case 5:e=g.i,c=(f=e)==null?void 0:f.config;case 4:return g.yield(Iq(c,b,d),3);case 3:if(c)for(var h=c,k=y(a.i.h),l=k.next();!l.done;l=k.next())l=l.value,l(h);g.h=0}})}
function Qq(a,b,c){var d,e,f,g;return B(function(h){if(h.h==1){if(!T("start_client_gcf"))return h.B(0);a.coldHashData=b;E("yt.gcf.config.coldHashData",a.coldHashData||null);return(d=pq())?c?h.B(4):h.yield(Kq(d),5):h.B(0)}h.h!=4&&(e=h.i,c=(f=e)==null?void 0:f.config);if(!c)return h.B(0);g=c.configData;return h.yield(Jq(c,b,g,d),0)})}
function Rq(){if(!Nq.instance){var a=new Nq;Nq.instance=a}a=Nq.instance;var b=V()-a.h;if(!(a.h!==0&&b<ln("send_config_hash_timer"))){b=F("yt.gcf.config.coldConfigData");var c=F("yt.gcf.config.hotHashData"),d=F("yt.gcf.config.coldHashData");b&&c&&d&&(a.h=V());return{coldConfigData:b,hotHashData:c,coldHashData:d}}}
Nq.prototype.o=function(a){this.hotHashData=a;E("yt.gcf.config.hotHashData",this.hotHashData||null)};function Sq(){return"INNERTUBE_API_KEY"in Km&&"INNERTUBE_API_VERSION"in Km}
function Tq(){return{innertubeApiKey:R("INNERTUBE_API_KEY"),innertubeApiVersion:R("INNERTUBE_API_VERSION"),Ae:R("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),yd:R("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),Nh:R("INNERTUBE_CONTEXT_CLIENT_NAME",1),innertubeContextClientVersion:R("INNERTUBE_CONTEXT_CLIENT_VERSION"),Ce:R("INNERTUBE_CONTEXT_HL"),Be:R("INNERTUBE_CONTEXT_GL"),De:R("INNERTUBE_HOST_OVERRIDE")||"",Ee:!!R("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),Oh:!!R("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",
!1),appInstallData:R("SERIALIZED_CLIENT_CONFIG_DATA")}}
function Uq(a){var b={client:{hl:a.Ce,gl:a.Be,clientName:a.yd,clientVersion:a.innertubeContextClientVersion,configInfo:a.Ae}};navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=D.devicePixelRatio;c&&c!=1&&(b.client.screenDensityFloat=String(c));c=R("EXPERIMENTS_TOKEN","");c!==""&&(b.client.experimentsToken=c);c=mn();c.length>0&&(b.request={internalExperimentFlags:c});c=a.yd;if((c==="WEB"||c==="MWEB"||c===1||c===2)&&b){var d;b.client.mainAppWebInfo=(d=b.client.mainAppWebInfo)!=
null?d:{};b.client.mainAppWebInfo.webDisplayMode=Qn()}(d=F("yt.embedded_player.embed_url"))&&b&&(b.thirdParty={embedUrl:d});var e;if(T("web_log_memory_total_kbytes")&&((e=D.navigator)==null?0:e.deviceMemory)){var f;e=(f=D.navigator)==null?void 0:f.deviceMemory;b&&(b.client.memoryTotalKbytes=""+e*1E6)}a.appInstallData&&b&&(b.client.configInfo=b.client.configInfo||{},b.client.configInfo.appInstallData=a.appInstallData);(a=ko())&&b&&(b.client.connectionType=a);T("web_log_effective_connection_type")&&
(a=lo())&&b&&(b.client.effectiveConnectionType=a);T("start_client_gcf")&&(e=Rq())&&(a=e.coldConfigData,f=e.coldHashData,e=e.hotHashData,b&&(b.client.configInfo=b.client.configInfo||{},a&&(b.client.configInfo.coldConfigData=a),f&&(b.client.configInfo.coldHashData=f),e&&(b.client.configInfo.hotHashData=e)));R("DELEGATED_SESSION_ID")&&!T("pageid_as_header_web")&&(b.user={onBehalfOfUser:R("DELEGATED_SESSION_ID")});!T("fill_delegate_context_in_gel_killswitch")&&(a=R("INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT"))&&
(b.user=Object.assign({},b.user,{serializedDelegationContext:a}));a=R("INNERTUBE_CONTEXT");var g;if(T("enable_persistent_device_token")&&(a==null?0:(g=a.client)==null?0:g.rolloutToken)){var h;b.client.rolloutToken=a==null?void 0:(h=a.client)==null?void 0:h.rolloutToken}g=Object;h=g.assign;a=b.client;f={};e=y(Object.entries(Ym(R("DEVICE",""))));for(d=e.next();!d.done;d=e.next())c=y(d.value),d=c.next().value,c=c.next().value,d==="cbrand"?f.deviceMake=c:d==="cmodel"?f.deviceModel=c:d==="cbr"?f.browserName=
c:d==="cbrver"?f.browserVersion=c:d==="cos"?f.osName=c:d==="cosver"?f.osVersion=c:d==="cplatform"&&(f.platform=c);b.client=h.call(g,a,f);return b}
function Vq(a,b,c){c=c===void 0?{}:c;var d={};R("EOM_VISITOR_DATA")?d={"X-Goog-EOM-Visitor-Id":R("EOM_VISITOR_DATA")}:d={"X-Goog-Visitor-Id":c.visitorData||R("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;b=c.authorization||R("AUTHORIZATION");b||(a?b="Bearer "+F("gapi.auth.getToken")().th:(a=Tn(Sn()),T("pageid_as_header_web")||delete a["X-Goog-PageId"],d=Object.assign({},d,a)));b&&(d.Authorization=b);return d}
;var Wq=typeof TextEncoder!=="undefined"?new TextEncoder:null,Xq=Wq?function(a){return Wq.encode(a)}:function(a){for(var b=[],c=0,d=0;d<a.length;d++){var e=a.charCodeAt(d);
e<128?b[c++]=e:(e<2048?b[c++]=e>>6|192:((e&64512)==55296&&d+1<a.length&&(a.charCodeAt(d+1)&64512)==56320?(e=65536+((e&1023)<<10)+(a.charCodeAt(++d)&1023),b[c++]=e>>18|240,b[c++]=e>>12&63|128):b[c++]=e>>12|224,b[c++]=e>>6&63|128),b[c++]=e&63|128)}a=new Uint8Array(b.length);for(c=0;c<a.length;c++)a[c]=b[c];return a};var Yq={next:"wn_s",browse:"br_s",search:"sr_s",reel:"r_wrs",player:"ps_s"},Zq={next:"wn_r",browse:"br_r",search:"sr_r",reel:"r_wrr",player:"ps_r"};function $q(a,b){this.version=a;this.args=b}
$q.prototype.serialize=function(){return{version:this.version,args:this.args}};function ar(a,b){this.topic=a;this.h=b}
ar.prototype.toString=function(){return this.topic};var br=F("ytPubsub2Pubsub2Instance")||new N;N.prototype.subscribe=N.prototype.subscribe;N.prototype.unsubscribeByKey=N.prototype.fc;N.prototype.publish=N.prototype.sb;N.prototype.clear=N.prototype.clear;E("ytPubsub2Pubsub2Instance",br);var cr=F("ytPubsub2Pubsub2SubscribedKeys")||{};E("ytPubsub2Pubsub2SubscribedKeys",cr);var dr=F("ytPubsub2Pubsub2TopicToKeys")||{};E("ytPubsub2Pubsub2TopicToKeys",dr);var er=F("ytPubsub2Pubsub2IsAsync")||{};E("ytPubsub2Pubsub2IsAsync",er);
E("ytPubsub2Pubsub2SkipSubKey",null);function fr(a,b){var c=gr();c&&c.publish.call(c,a.toString(),a,b)}
function hr(a){var b=ir,c=gr();if(!c)return 0;var d=c.subscribe(b.toString(),function(e,f){var g=F("ytPubsub2Pubsub2SkipSubKey");g&&g==d||(g=function(){if(cr[d])try{if(f&&b instanceof ar&&b!=e)try{var h=b.h,k=f;if(!k.args||!k.version)throw Error("yt.pubsub2.Data.deserialize(): serializedData is incomplete.");try{if(!h.Qd){var l=new h;h.Qd=l.version}var m=h.Qd}catch(n){}if(!m||k.version!=m)throw Error("yt.pubsub2.Data.deserialize(): serializedData version is incompatible.");try{f=Reflect.construct(h,
Zb(k.args))}catch(n){throw n.message="yt.pubsub2.Data.deserialize(): "+n.message,n;}}catch(n){throw n.message="yt.pubsub2.pubsub2 cross-binary conversion error for "+b.toString()+": "+n.message,n;}a.call(window,f)}catch(n){Qm(n)}},er[b.toString()]?F("yt.scheduler.instance")?Xj.pa(g):hn(g,0):g())});
cr[d]=!0;dr[b.toString()]||(dr[b.toString()]=[]);dr[b.toString()].push(d);return d}
function jr(){var a=kr,b=hr(function(c){a.apply(void 0,arguments);lr(b)});
return b}
function lr(a){var b=gr();b&&(typeof a==="number"&&(a=[a]),Tb(a,function(c){b.unsubscribeByKey(c);delete cr[c]}))}
function gr(){return F("ytPubsub2Pubsub2Instance")}
;function mr(a,b,c){c=c===void 0?{sampleRate:.1}:c;Math.random()<Math.min(.02,c.sampleRate/100)&&fr("meta_logging_csi_event",{timerName:a,ki:b})}
;var nr=void 0,or=void 0;function pr(){or||(or=hm(R("WORKER_SERIALIZATION_URL")));return or||void 0}
function qr(){var a=pr();nr||a===void 0||(nr=new Worker(pb(a),void 0));return nr}
;var rr=ln("max_body_size_to_compress",5E5),sr=ln("min_body_size_to_compress",500),tr=!0,ur=0,vr=0,wr=ln("compression_performance_threshold_lr",250),xr=ln("slow_compressions_before_abandon_count",4),yr=!1,zr=new Map,Ar=1,Br=!0;function Cr(){if(typeof Worker==="function"&&pr()&&!yr){var a=function(c){c=c.data;if(c.op==="gzippedGelBatch"){var d=zr.get(c.key);d&&(Dr(c.gzippedBatch,d.latencyPayload,d.url,d.options,d.sendFn),zr.delete(c.key))}},b=qr();
b&&(b.addEventListener("message",a),b.onerror=function(){zr.clear()},yr=!0)}}
function Er(a,b,c,d,e){e=e===void 0?!1:e;var f={startTime:V(),ticks:{},infos:{}};if(tr)try{var g=Fr(b);if(g!=null&&(g>rr||g<sr))d(a,c);else{if(T("gzip_gel_with_worker")&&(T("initial_gzip_use_main_thread")&&!Br||!T("initial_gzip_use_main_thread"))){yr||Cr();var h=qr();if(h&&!e){zr.set(Ar,{latencyPayload:f,url:a,options:c,sendFn:d});h.postMessage({op:"gelBatchToGzip",serializedBatch:b,key:Ar});Ar++;return}}var k=fm(Xq(b));Dr(k,f,a,c,d)}}catch(l){Rm(l),d(a,c)}else d(a,c)}
function Dr(a,b,c,d,e){Br=!1;var f=V();b.ticks.gelc=f;vr++;T("disable_compression_due_to_performance_degredation")&&f-b.startTime>=wr&&(ur++,T("abandon_compression_after_N_slow_zips")?vr===ln("compression_disable_point")&&ur>xr&&(tr=!1):tr=!1);Gr(b);d.headers||(d.headers={});d.headers["Content-Encoding"]="gzip";d.postBody=a;d.postParams=void 0;e(c,d)}
function Hr(a){var b=b===void 0?!1:b;var c=c===void 0?!1:c;var d=V(),e={startTime:d,ticks:{},infos:{}},f=b?F("yt.logging.gzipForFetch",!1):!0;if(tr&&f){if(!a.body)return a;try{var g=c?a.body:typeof a.body==="string"?a.body:JSON.stringify(a.body);f=g;if(!c&&typeof g==="string"){var h=Fr(g);if(h!=null&&(h>rr||h<sr))return a;c=b?{level:1}:void 0;f=fm(Xq(g),c);var k=V();e.ticks.gelc=k;if(b){vr++;if((T("disable_compression_due_to_performance_degredation")||T("disable_compression_due_to_performance_degradation_lr"))&&
k-d>=wr)if(ur++,T("abandon_compression_after_N_slow_zips")||T("abandon_compression_after_N_slow_zips_lr")){b=ur/vr;var l=xr/ln("compression_disable_point");vr>0&&vr%ln("compression_disable_point")===0&&b>=l&&(tr=!1)}else tr=!1;Gr(e)}}a.headers=Object.assign({},{"Content-Encoding":"gzip"},a.headers||{});a.body=f;return a}catch(m){return Rm(m),a}}else return a}
function Fr(a){try{return(new Blob(a.split(""))).size}catch(b){return Rm(b),null}}
function Gr(a){T("gel_compression_csi_killswitch")||!T("log_gel_compression_latency")&&!T("log_gel_compression_latency_lr")||mr("gel_compression",a,{sampleRate:.1})}
;function Ir(a){a=Object.assign({},a);delete a.Authorization;var b=mg();if(b){var c=new bk;c.update(R("INNERTUBE_API_KEY"));c.update(b);a.hash=vd(c.digest(),3)}return a}
;var Jr;function Kr(){Jr||(Jr=new Vo("yt.innertube"));return Jr}
function Lr(a,b,c,d){if(d)return null;d=Kr().get("nextId",!0)||1;var e=Kr().get("requests",!0)||{};e[d]={method:a,request:b,authState:Ir(c),requestTime:Math.round(V())};Kr().set("nextId",d+1,86400,!0);Kr().set("requests",e,86400,!0);return d}
function Mr(a){var b=Kr().get("requests",!0)||{};delete b[a];Kr().set("requests",b,86400,!0)}
function Nr(a){var b=Kr().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(Math.round(V())-d.requestTime<6E4)){var e=d.authState,f=Ir(Vq(!1));Bg(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(V())),Or(a,d.method,e,{}));delete b[c]}}Kr().set("requests",b,86400,!0)}}
;function Pr(a){this.ic=this.h=!1;this.potentialEsfErrorCounter=this.i=0;this.handleError=function(){};
this.zb=function(){};
this.now=Date.now;this.Sb=!1;var b;this.Od=(b=a.Od)!=null?b:100;var c;this.Jd=(c=a.Jd)!=null?c:1;var d;this.Hd=(d=a.Hd)!=null?d:2592E6;var e;this.Gd=(e=a.Gd)!=null?e:12E4;var f;this.Id=(f=a.Id)!=null?f:5E3;var g;this.V=(g=a.V)!=null?g:void 0;this.oc=!!a.oc;var h;this.lc=(h=a.lc)!=null?h:.1;var k;this.Bc=(k=a.Bc)!=null?k:10;a.handleError&&(this.handleError=a.handleError);a.zb&&(this.zb=a.zb);a.Sb&&(this.Sb=a.Sb);a.ic&&(this.ic=a.ic);this.W=a.W;this.Ca=a.Ca;this.ga=a.ga;this.fa=a.fa;this.sendFn=a.sendFn;
this.Yc=a.Yc;this.Vc=a.Vc;Qr(this)&&(!this.W||this.W("networkless_logging"))&&Rr(this)}
function Rr(a){Qr(a)&&!a.Sb&&(a.h=!0,a.oc&&Math.random()<=a.lc&&a.ga.de(a.V),Sr(a),a.fa.ta()&&a.ec(),a.fa.listen(a.Yc,a.ec.bind(a)),a.fa.listen(a.Vc,a.rd.bind(a)))}
p=Pr.prototype;p.writeThenSend=function(a,b){var c=this;b=b===void 0?{}:b;if(Qr(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.ga.set(d,this.V).then(function(e){d.id=e;c.fa.ta()&&Tr(c,d)}).catch(function(e){Tr(c,d);
Ur(c,e)})}else this.sendFn(a,b)};
p.sendThenWrite=function(a,b,c){var d=this;b=b===void 0?{}:b;if(Qr(this)&&this.h){var e={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.W&&this.W("nwl_skip_retry")&&(e.skipRetry=c);if(this.fa.ta()||this.W&&this.W("nwl_aggressive_send_then_write")&&!e.skipRetry){if(!e.skipRetry){var f=b.onError?b.onError:function(){};
b.onError=function(g,h){return B(function(k){if(k.h==1)return k.yield(d.ga.set(e,d.V).catch(function(l){Ur(d,l)}),2);
f(g,h);k.h=0})}}this.sendFn(a,b,e.skipRetry)}else this.ga.set(e,this.V).catch(function(g){d.sendFn(a,b,e.skipRetry);
Ur(d,g)})}else this.sendFn(a,b,this.W&&this.W("nwl_skip_retry")&&c)};
p.sendAndWrite=function(a,b){var c=this;b=b===void 0?{}:b;if(Qr(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0},e=!1,f=b.onSuccess?b.onSuccess:function(){};
d.options.onSuccess=function(g,h){d.id!==void 0?c.ga.xb(d.id,c.V):e=!0;c.fa.mb&&c.W&&c.W("vss_network_hint")&&c.fa.mb(!0);f(g,h)};
this.sendFn(d.url,d.options,void 0,!0);this.ga.set(d,this.V).then(function(g){d.id=g;e&&c.ga.xb(d.id,c.V)}).catch(function(g){Ur(c,g)})}else this.sendFn(a,b,void 0,!0)};
p.ec=function(){var a=this;if(!Qr(this))throw Error("IndexedDB is not supported: throttleSend");this.i||(this.i=this.Ca.pa(function(){var b;return B(function(c){if(c.h==1)return c.yield(a.ga.vd("NEW",a.V),2);if(c.h!=3)return b=c.i,b?c.yield(Tr(a,b),3):(a.rd(),c.return());a.i&&(a.i=0,a.ec());c.h=0})},this.Od))};
p.rd=function(){this.Ca.qa(this.i);this.i=0};
function Tr(a,b){var c;return B(function(d){switch(d.h){case 1:if(!Qr(a))throw Error("IndexedDB is not supported: immediateSend");if(b.id===void 0){d.B(2);break}return d.yield(a.ga.Ie(b.id,a.V),3);case 3:(c=d.i)||a.zb(Error("The request cannot be found in the database."));case 2:if(Vr(a,b,a.Hd)){d.B(4);break}a.zb(Error("Networkless Logging: Stored logs request expired age limit"));if(b.id===void 0){d.B(5);break}return d.yield(a.ga.xb(b.id,a.V),5);case 5:return d.return();case 4:b.skipRetry||(b=Wr(a,
b));if(!b){d.B(0);break}if(!b.skipRetry||b.id===void 0){d.B(8);break}return d.yield(a.ga.xb(b.id,a.V),8);case 8:a.sendFn(b.url,b.options,!!b.skipRetry),d.h=0}})}
function Wr(a,b){if(!Qr(a))throw Error("IndexedDB is not supported: updateRequestHandlers");var c=b.options.onError?b.options.onError:function(){};
b.options.onError=function(e,f){var g,h,k,l;return B(function(m){switch(m.h){case 1:g=Xr(f);(h=Yr(f))&&a.W&&a.W("web_enable_error_204")&&a.handleError(Error("Request failed due to compression"),b.url,f);if(!(a.W&&a.W("nwl_consider_error_code")&&g||a.W&&!a.W("nwl_consider_error_code")&&a.potentialEsfErrorCounter<=a.Bc)){m.B(2);break}if(!a.fa.Fc){m.B(3);break}return m.yield(a.fa.Fc(),3);case 3:if(a.fa.ta()){m.B(2);break}c(e,f);if(!a.W||!a.W("nwl_consider_error_code")||((k=b)==null?void 0:k.id)===void 0){m.B(6);
break}return m.yield(a.ga.Zc(b.id,a.V,!1),6);case 6:return m.return();case 2:if(a.W&&a.W("nwl_consider_error_code")&&!g&&a.potentialEsfErrorCounter>a.Bc)return m.return();a.potentialEsfErrorCounter++;if(((l=b)==null?void 0:l.id)===void 0){m.B(8);break}return b.sendCount<a.Jd?m.yield(a.ga.Zc(b.id,a.V,!0,h?!1:void 0),12):m.yield(a.ga.xb(b.id,a.V),8);case 12:a.Ca.pa(function(){a.fa.ta()&&a.ec()},a.Id);
case 8:c(e,f),m.h=0}})};
var d=b.options.onSuccess?b.options.onSuccess:function(){};
b.options.onSuccess=function(e,f){var g;return B(function(h){if(h.h==1)return((g=b)==null?void 0:g.id)===void 0?h.B(2):h.yield(a.ga.xb(b.id,a.V),2);a.fa.mb&&a.W&&a.W("vss_network_hint")&&a.fa.mb(!0);d(e,f);h.h=0})};
return b}
function Vr(a,b,c){b=b.timestamp;return a.now()-b>=c?!1:!0}
function Sr(a){if(!Qr(a))throw Error("IndexedDB is not supported: retryQueuedRequests");a.ga.vd("QUEUED",a.V).then(function(b){b&&!Vr(a,b,a.Gd)?a.Ca.pa(function(){return B(function(c){if(c.h==1)return b.id===void 0?c.B(2):c.yield(a.ga.Zc(b.id,a.V),2);Sr(a);c.h=0})}):a.fa.ta()&&a.ec()})}
function Ur(a,b){a.Td&&!a.fa.ta()?a.Td(b):a.handleError(b)}
function Qr(a){return!!a.V||a.ic}
function Xr(a){var b;return(a=a==null?void 0:(b=a.error)==null?void 0:b.code)&&a>=400&&a<=599?!1:!0}
function Yr(a){var b;a=a==null?void 0:(b=a.error)==null?void 0:b.code;return!(a!==400&&a!==415)}
;var Zr;
function $r(){if(Zr)return Zr();var a={};Zr=Eq("LogsDatabaseV2",{Hb:(a.LogsRequestsStore={Ob:2},a),shared:!1,upgrade:function(b,c,d){c(2)&&Kp(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0});c(3);c(5)&&(d=d.objectStore("LogsRequestsStore"),d.h.indexNames.contains("newRequest")&&d.h.deleteIndex("newRequest"),Rp(d,"newRequestV2",["status","interface","timestamp"]));c(7)&&b.h.objectStoreNames.contains("sapisid")&&b.h.deleteObjectStore("sapisid");c(9)&&b.h.objectStoreNames.contains("SWHealthLog")&&b.h.deleteObjectStore("SWHealthLog")},
version:9});return Zr()}
;function as(a){return dq($r(),a)}
function bs(a,b){var c,d,e,f;return B(function(g){if(g.h==1)return c={startTime:V(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},ticks:{}},g.yield(as(b),2);if(g.h!=3)return d=g.i,e=Object.assign({},a,{options:JSON.parse(JSON.stringify(a.options)),interface:R("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),g.yield(Mp(d,"LogsRequestsStore",e),3);f=g.i;c.ticks.tc=V();cs(c);return g.return(f)})}
function ds(a,b){var c,d,e,f,g,h,k,l;return B(function(m){if(m.h==1)return c={startTime:V(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},ticks:{}},m.yield(as(b),2);if(m.h!=3)return d=m.i,e=R("INNERTUBE_CONTEXT_CLIENT_NAME",0),f=[a,e,0],g=[a,e,V()],h=IDBKeyRange.bound(f,g),k="prev",T("use_fifo_for_networkless")&&(k="next"),l=void 0,m.yield(Jp(d,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(n){return Xp(n.objectStore("LogsRequestsStore").index("newRequestV2"),{query:h,direction:k},
function(r){r.getValue()&&(l=r.getValue(),a==="NEW"&&(l.status="QUEUED",r.update(l)))})}),3);
c.ticks.tc=V();cs(c);return m.return(l)})}
function es(a,b){var c;return B(function(d){if(d.h==1)return d.yield(as(b),2);c=d.i;return d.return(Jp(c,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(e){var f=e.objectStore("LogsRequestsStore");return f.get(a).then(function(g){if(g)return g.status="QUEUED",Fp(f.h.put(g,void 0)).then(function(){return g})})}))})}
function gs(a,b,c,d){c=c===void 0?!0:c;var e;return B(function(f){if(f.h==1)return f.yield(as(b),2);e=f.i;return f.return(Jp(e,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(g){var h=g.objectStore("LogsRequestsStore");return h.get(a).then(function(k){return k?(k.status="NEW",c&&(k.sendCount+=1),d!==void 0&&(k.options.compress=d),Fp(h.h.put(k,void 0)).then(function(){return k})):zp.resolve(void 0)})}))})}
function hs(a,b){var c;return B(function(d){if(d.h==1)return d.yield(as(b),2);c=d.i;return d.return(c.delete("LogsRequestsStore",a))})}
function is(a){var b,c;return B(function(d){if(d.h==1)return d.yield(as(a),2);b=d.i;c=V()-2592E6;return d.yield(Jp(b,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(e){return Tp(e.objectStore("LogsRequestsStore"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return Up(f)})})}),0)})}
function js(){B(function(a){return a.yield(Aq(),0)})}
function cs(a){T("nwl_csi_killswitch")||mr("networkless_performance",a,{sampleRate:1})}
;var ks={accountStateChangeSignedIn:23,accountStateChangeSignedOut:24,delayedEventMetricCaptured:11,latencyActionBaselined:6,latencyActionInfo:7,latencyActionTicked:5,offlineTransferStatusChanged:2,offlineImageDownload:335,playbackStartStateChanged:9,systemHealthCaptured:3,mangoOnboardingCompleted:10,mangoPushNotificationReceived:230,mangoUnforkDbMigrationError:121,mangoUnforkDbMigrationSummary:122,mangoUnforkDbMigrationPreunforkDbVersionNumber:133,mangoUnforkDbMigrationPhoneMetadata:134,mangoUnforkDbMigrationPhoneStorage:135,
mangoUnforkDbMigrationStep:142,mangoAsyncApiMigrationEvent:223,mangoDownloadVideoResult:224,mangoHomepageVideoCount:279,mangoHomeV3State:295,mangoImageClientCacheHitEvent:273,sdCardStatusChanged:98,framesDropped:12,thumbnailHovered:13,deviceRetentionInfoCaptured:14,thumbnailLoaded:15,backToAppEvent:318,streamingStatsCaptured:17,offlineVideoShared:19,appCrashed:20,youThere:21,offlineStateSnapshot:22,mdxSessionStarted:25,mdxSessionConnected:26,mdxSessionDisconnected:27,bedrockResourceConsumptionSnapshot:28,
nextGenWatchWatchSwiped:29,kidsAccountsSnapshot:30,zeroStepChannelCreated:31,tvhtml5SearchCompleted:32,offlineSharePairing:34,offlineShareUnlock:35,mdxRouteDistributionSnapshot:36,bedrockRepetitiveActionTimed:37,unpluggedDegradationInfo:229,uploadMp4HeaderMoved:38,uploadVideoTranscoded:39,uploadProcessorStarted:46,uploadProcessorEnded:47,uploadProcessorReady:94,uploadProcessorRequirementPending:95,uploadProcessorInterrupted:96,uploadFrontendEvent:241,assetPackDownloadStarted:41,assetPackDownloaded:42,
assetPackApplied:43,assetPackDeleted:44,appInstallAttributionEvent:459,playbackSessionStopped:45,adBlockerMessagingShown:48,distributionChannelCaptured:49,dataPlanCpidRequested:51,detailedNetworkTypeCaptured:52,sendStateUpdated:53,receiveStateUpdated:54,sendDebugStateUpdated:55,receiveDebugStateUpdated:56,kidsErrored:57,mdxMsnSessionStatsFinished:58,appSettingsCaptured:59,mdxWebSocketServerHttpError:60,mdxWebSocketServer:61,startupCrashesDetected:62,coldStartInfo:435,offlinePlaybackStarted:63,liveChatMessageSent:225,
liveChatUserPresent:434,liveChatBeingModerated:457,liveCreationCameraUpdated:64,liveCreationEncodingCaptured:65,liveCreationError:66,liveCreationHealthUpdated:67,liveCreationVideoEffectsCaptured:68,liveCreationStageOccured:75,liveCreationBroadcastScheduled:123,liveCreationArchiveReplacement:149,liveCreationCostreamingConnection:421,liveCreationStreamWebrtcStats:288,mdxSessionRecoveryStarted:69,mdxSessionRecoveryCompleted:70,mdxSessionRecoveryStopped:71,visualElementShown:72,visualElementHidden:73,
visualElementGestured:78,visualElementStateChanged:208,screenCreated:156,playbackAssociated:202,visualElementAttached:215,playbackContextEvent:214,cloudCastingPlaybackStarted:74,webPlayerApiCalled:76,tvhtml5AccountDialogOpened:79,foregroundHeartbeat:80,foregroundHeartbeatScreenAssociated:111,kidsOfflineSnapshot:81,mdxEncryptionSessionStatsFinished:82,playerRequestCompleted:83,liteSchedulerStatistics:84,mdxSignIn:85,spacecastMetadataLookupRequested:86,spacecastBatchLookupRequested:87,spacecastSummaryRequested:88,
spacecastPlayback:89,spacecastDiscovery:90,tvhtml5LaunchUrlComponentChanged:91,mdxBackgroundPlaybackRequestCompleted:92,mdxBrokenAdditionalDataDeviceDetected:93,tvhtml5LocalStorage:97,tvhtml5DeviceStorageStatus:147,autoCaptionsAvailable:99,playbackScrubbingEvent:339,flexyState:100,interfaceOrientationCaptured:101,mainAppBrowseFragmentCache:102,offlineCacheVerificationFailure:103,offlinePlaybackExceptionDigest:217,vrCopresenceStats:104,vrCopresenceSyncStats:130,vrCopresenceCommsStats:137,vrCopresencePartyStats:153,
vrCopresenceEmojiStats:213,vrCopresenceEvent:141,vrCopresenceFlowTransitEvent:160,vrCowatchPartyEvent:492,vrCowatchUserStartOrJoinEvent:504,vrPlaybackEvent:345,kidsAgeGateTracking:105,offlineDelayAllowedTracking:106,mainAppAutoOfflineState:107,videoAsThumbnailDownload:108,videoAsThumbnailPlayback:109,liteShowMore:110,renderingError:118,kidsProfilePinGateTracking:119,abrTrajectory:124,scrollEvent:125,streamzIncremented:126,kidsProfileSwitcherTracking:127,kidsProfileCreationTracking:129,buyFlowStarted:136,
mbsConnectionInitiated:138,mbsPlaybackInitiated:139,mbsLoadChildren:140,liteProfileFetcher:144,mdxRemoteTransaction:146,reelPlaybackError:148,reachabilityDetectionEvent:150,mobilePlaybackEvent:151,courtsidePlayerStateChanged:152,musicPersistentCacheChecked:154,musicPersistentCacheCleared:155,playbackInterrupted:157,playbackInterruptionResolved:158,fixFopFlow:159,anrDetection:161,backstagePostCreationFlowEnded:162,clientError:163,gamingAccountLinkStatusChanged:164,liteHousewarming:165,buyFlowEvent:167,
kidsParentalGateTracking:168,kidsSignedOutSettingsStatus:437,kidsSignedOutPauseHistoryFixStatus:438,tvhtml5WatchdogViolation:444,ypcUpgradeFlow:169,yongleStudy:170,ypcUpdateFlowStarted:171,ypcUpdateFlowCancelled:172,ypcUpdateFlowSucceeded:173,ypcUpdateFlowFailed:174,liteGrowthkitPromo:175,paymentFlowStarted:341,transactionFlowShowPaymentDialog:405,transactionFlowStarted:176,transactionFlowSecondaryDeviceStarted:222,transactionFlowSecondaryDeviceSignedOutStarted:383,transactionFlowCancelled:177,transactionFlowPaymentCallBackReceived:387,
transactionFlowPaymentSubmitted:460,transactionFlowPaymentSucceeded:329,transactionFlowSucceeded:178,transactionFlowFailed:179,transactionFlowPlayBillingConnectionStartEvent:428,transactionFlowSecondaryDeviceSuccess:458,transactionFlowErrorEvent:411,liteVideoQualityChanged:180,watchBreakEnablementSettingEvent:181,watchBreakFrequencySettingEvent:182,videoEffectsCameraPerformanceMetrics:183,adNotify:184,startupTelemetry:185,playbackOfflineFallbackUsed:186,outOfMemory:187,ypcPauseFlowStarted:188,ypcPauseFlowCancelled:189,
ypcPauseFlowSucceeded:190,ypcPauseFlowFailed:191,uploadFileSelected:192,ypcResumeFlowStarted:193,ypcResumeFlowCancelled:194,ypcResumeFlowSucceeded:195,ypcResumeFlowFailed:196,adsClientStateChange:197,ypcCancelFlowStarted:198,ypcCancelFlowCancelled:199,ypcCancelFlowSucceeded:200,ypcCancelFlowFailed:201,ypcCancelFlowGoToPaymentProcessor:402,ypcDeactivateFlowStarted:320,ypcRedeemFlowStarted:203,ypcRedeemFlowCancelled:204,ypcRedeemFlowSucceeded:205,ypcRedeemFlowFailed:206,ypcFamilyCreateFlowStarted:258,
ypcFamilyCreateFlowCancelled:259,ypcFamilyCreateFlowSucceeded:260,ypcFamilyCreateFlowFailed:261,ypcFamilyManageFlowStarted:262,ypcFamilyManageFlowCancelled:263,ypcFamilyManageFlowSucceeded:264,ypcFamilyManageFlowFailed:265,restoreContextEvent:207,embedsAdEvent:327,autoplayTriggered:209,clientDataErrorEvent:210,experimentalVssValidation:211,tvhtml5TriggeredEvent:212,tvhtml5FrameworksFieldTrialResult:216,tvhtml5FrameworksFieldTrialStart:220,musicOfflinePreferences:218,watchTimeSegment:219,appWidthLayoutError:221,
accountRegistryChange:226,userMentionAutoCompleteBoxEvent:227,downloadRecommendationEnablementSettingEvent:228,musicPlaybackContentModeChangeEvent:231,offlineDbOpenCompleted:232,kidsFlowEvent:233,kidsFlowCorpusSelectedEvent:234,videoEffectsEvent:235,unpluggedOpsEogAnalyticsEvent:236,playbackAudioRouteEvent:237,interactionLoggingDebugModeError:238,offlineYtbRefreshed:239,kidsFlowError:240,musicAutoplayOnLaunchAttempted:242,deviceContextActivityEvent:243,deviceContextEvent:244,templateResolutionException:245,
musicSideloadedPlaylistServiceCalled:246,embedsStorageAccessNotChecked:247,embedsHasStorageAccessResult:248,embedsItpPlayedOnReload:249,embedsRequestStorageAccessResult:250,embedsShouldRequestStorageAccessResult:251,embedsRequestStorageAccessState:256,embedsRequestStorageAccessFailedState:257,embedsItpWatchLaterResult:266,searchSuggestDecodingPayloadFailure:252,siriShortcutActivated:253,tvhtml5KeyboardPerformance:254,latencyActionSpan:255,elementsLog:267,ytbFileOpened:268,tfliteModelError:269,apiTest:270,
yongleUsbSetup:271,touStrikeInterstitialEvent:272,liteStreamToSave:274,appBundleClientEvent:275,ytbFileCreationFailed:276,adNotifyFailure:278,ytbTransferFailed:280,blockingRequestFailed:281,liteAccountSelector:282,liteAccountUiCallbacks:283,dummyPayload:284,browseResponseValidationEvent:285,entitiesError:286,musicIosBackgroundFetch:287,mdxNotificationEvent:289,layersValidationError:290,musicPwaInstalled:291,liteAccountCleanup:292,html5PlayerHealthEvent:293,watchRestoreAttempt:294,liteAccountSignIn:296,
notaireEvent:298,kidsVoiceSearchEvent:299,adNotifyFilled:300,delayedEventDropped:301,analyticsSearchEvent:302,systemDarkThemeOptOutEvent:303,flowEvent:304,networkConnectivityBaselineEvent:305,ytbFileImported:306,downloadStreamUrlExpired:307,directSignInEvent:308,lyricImpressionEvent:309,accessibilityStateEvent:310,tokenRefreshEvent:311,genericAttestationExecution:312,tvhtml5VideoSeek:313,unpluggedAutoPause:314,scrubbingEvent:315,bedtimeReminderEvent:317,tvhtml5UnexpectedRestart:319,tvhtml5StabilityTraceEvent:478,
tvhtml5OperationHealth:467,tvhtml5WatchKeyEvent:321,voiceLanguageChanged:322,tvhtml5LiveChatStatus:323,parentToolsCorpusSelectedEvent:324,offerAdsEnrollmentInitiated:325,networkQualityIntervalEvent:326,deviceStartupMetrics:328,heartbeatActionPlayerTransitioned:330,tvhtml5Lifecycle:331,heartbeatActionPlayerHalted:332,adaptiveInlineMutedSettingEvent:333,mainAppLibraryLoadingState:334,thirdPartyLogMonitoringEvent:336,appShellAssetLoadReport:337,tvhtml5AndroidAttestation:338,tvhtml5StartupSoundEvent:340,
iosBackgroundRefreshTask:342,iosBackgroundProcessingTask:343,sliEventBatch:344,postImpressionEvent:346,musicSideloadedPlaylistExport:347,idbUnexpectedlyClosed:348,voiceSearchEvent:349,mdxSessionCastEvent:350,idbQuotaExceeded:351,idbTransactionEnded:352,idbTransactionAborted:353,tvhtml5KeyboardLogging:354,idbIsSupportedCompleted:355,creatorStudioMobileEvent:356,idbDataCorrupted:357,parentToolsAppChosenEvent:358,webViewBottomSheetResized:359,activeStateControllerScrollPerformanceSummary:360,navigatorValidation:361,
mdxSessionHeartbeat:362,clientHintsPolyfillDiagnostics:363,clientHintsPolyfillEvent:364,proofOfOriginTokenError:365,kidsAddedAccountSummary:366,musicWearableDevice:367,ypcRefundFlowEvent:368,tvhtml5PlaybackMeasurementEvent:369,tvhtml5WatermarkMeasurementEvent:370,clientExpGcfPropagationEvent:371,mainAppReferrerIntent:372,leaderLockEnded:373,leaderLockAcquired:374,googleHatsEvent:375,persistentLensLaunchEvent:376,parentToolsChildWelcomeChosenEvent:378,browseThumbnailPreloadEvent:379,finalPayload:380,
mdxDialAdditionalDataUpdateEvent:381,webOrchestrationTaskLifecycleRecord:382,startupSignalEvent:384,accountError:385,gmsDeviceCheckEvent:386,accountSelectorEvent:388,accountUiCallbacks:389,mdxDialAdditionalDataProbeEvent:390,downloadsSearchIcingApiStats:391,downloadsSearchIndexUpdatedEvent:397,downloadsSearchIndexSnapshot:398,dataPushClientEvent:392,kidsCategorySelectedEvent:393,mdxDeviceManagementSnapshotEvent:394,prefetchRequested:395,prefetchableCommandExecuted:396,gelDebuggingEvent:399,webLinkTtsPlayEnd:400,
clipViewInvalid:401,persistentStorageStateChecked:403,cacheWipeoutEvent:404,playerEvent:410,sfvEffectPipelineStartedEvent:412,sfvEffectPipelinePausedEvent:429,sfvEffectPipelineEndedEvent:413,sfvEffectChosenEvent:414,sfvEffectLoadedEvent:415,sfvEffectUserInteractionEvent:465,sfvEffectFirstFrameProcessedLatencyEvent:416,sfvEffectAggregatedFramesProcessedLatencyEvent:417,sfvEffectAggregatedFramesDroppedEvent:418,sfvEffectPipelineErrorEvent:430,sfvEffectGraphFrozenEvent:419,sfvEffectGlThreadBlockedEvent:420,
mdeQosEvent:510,mdeVideoChangedEvent:442,mdePlayerPerformanceMetrics:472,mdeExporterEvent:497,genericClientExperimentEvent:423,homePreloadTaskScheduled:424,homePreloadTaskExecuted:425,homePreloadCacheHit:426,polymerPropertyChangedInObserver:427,applicationStarted:431,networkCronetRttBatch:432,networkCronetRttSummary:433,repeatChapterLoopEvent:436,seekCancellationEvent:462,lockModeTimeoutEvent:483,externalVideoShareToYoutubeAttempt:501,parentCodeEvent:502,offlineTransferStarted:4,musicOfflineMixtapePreferencesChanged:16,
mangoDailyNewVideosNotificationAttempt:40,mangoDailyNewVideosNotificationError:77,dtwsPlaybackStarted:112,dtwsTileFetchStarted:113,dtwsTileFetchCompleted:114,dtwsTileFetchStatusChanged:145,dtwsKeyframeDecoderBufferSent:115,dtwsTileUnderflowedOnNonkeyframe:116,dtwsBackfillFetchStatusChanged:143,dtwsBackfillUnderflowed:117,dtwsAdaptiveLevelChanged:128,blockingVisitorIdTimeout:277,liteSocial:18,mobileJsInvocation:297,biscottiBasedDetection:439,coWatchStateChange:440,embedsVideoDataDidChange:441,shortsFirst:443,
cruiseControlEvent:445,qoeClientLoggingContext:446,atvRecommendationJobExecuted:447,tvhtml5UserFeedback:448,producerProjectCreated:449,producerProjectOpened:450,producerProjectDeleted:451,producerProjectElementAdded:453,producerProjectElementRemoved:454,producerAppStateChange:509,producerProjectDiskInsufficientExportFailure:516,tvhtml5ShowClockEvent:455,deviceCapabilityCheckMetrics:456,youtubeClearcutEvent:461,offlineBrowseFallbackEvent:463,getCtvTokenEvent:464,startupDroppedFramesSummary:466,screenshotEvent:468,
miniAppPlayEvent:469,elementsDebugCounters:470,fontLoadEvent:471,webKillswitchReceived:473,webKillswitchExecuted:474,cameraOpenEvent:475,manualSmoothnessMeasurement:476,tvhtml5AppQualityEvent:477,polymerPropertyAccessEvent:479,miniAppSdkUsage:480,cobaltTelemetryEvent:481,crossDevicePlayback:482,channelCreatedWithObakeImage:484,channelEditedWithObakeImage:485,offlineDeleteEvent:486,crossDeviceNotificationTransfer:487,androidIntentEvent:488,unpluggedAmbientInterludesCounterfactualEvent:489,keyPlaysPlayback:490,
shortsCreationFallbackEvent:493,vssData:491,castMatch:494,miniAppPerformanceMetrics:495,userFeedbackEvent:496,kidsGuestSessionMismatch:498,musicSideloadedPlaylistMigrationEvent:499,sleepTimerSessionFinishEvent:500,watchEpPromoConflict:503,innertubeResponseCacheMetrics:505,miniAppAdEvent:506,dataPlanUpsellEvent:507,producerProjectRenamed:508,producerMediaSelectionEvent:511,embedsAutoplayStatusChanged:512,remoteConnectEvent:513,connectedSessionMisattributionEvent:514,producerProjectElementModified:515,
adsSeenClientLogging:517};var ls={},ms=Eq("ServiceWorkerLogsDatabase",{Hb:(ls.SWHealthLog={Ob:1},ls),shared:!0,upgrade:function(a,b){b(1)&&Rp(Kp(a,"SWHealthLog",{keyPath:"id",autoIncrement:!0}),"swHealthNewRequest",["interface","timestamp"])},
version:1});function ns(a){return dq(ms(),a)}
function ps(a){var b,c;B(function(d){if(d.h==1)return d.yield(ns(a),2);b=d.i;c=V()-2592E6;return d.yield(Jp(b,["SWHealthLog"],{mode:"readwrite",ka:!0},function(e){return Tp(e.objectStore("SWHealthLog"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return Up(f)})})}),0)})}
function qs(a){var b;return B(function(c){if(c.h==1)return c.yield(ns(a),2);b=c.i;return c.yield(b.clear("SWHealthLog"),0)})}
;var rs={},ss=0;function ts(a){var b=b===void 0?{}:b;var c=new Image,d=""+ss++;rs[d]=c;c.onload=c.onerror=function(){delete rs[d]};
b.gi&&(c.referrerPolicy="no-referrer");c.src=a}
;var us;function vs(){us||(us=new Vo("yt.offline"));return us}
function ws(a){if(T("offline_error_handling")){var b=vs().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);vs().set("errors",b,2592E3,!0)}}
;function xs(){this.h=new Map;this.i=!1}
function ys(){if(!xs.instance){var a=F("yt.networkRequestMonitor.instance")||new xs;E("yt.networkRequestMonitor.instance",a);xs.instance=a}return xs.instance}
xs.prototype.requestComplete=function(a,b){b&&(this.i=!0);a=this.removeParams(a);this.h.get(a)||this.h.set(a,b)};
xs.prototype.isEndpointCFR=function(a){a=this.removeParams(a);return(a=this.h.get(a))?!1:a===!1&&this.i?!0:null};
xs.prototype.removeParams=function(a){return a.split("?")[0]};
xs.prototype.removeParams=xs.prototype.removeParams;xs.prototype.isEndpointCFR=xs.prototype.isEndpointCFR;xs.prototype.requestComplete=xs.prototype.requestComplete;xs.getInstance=ys;function zs(){fi.call(this);var a=this;this.j=!1;this.h=Wj();this.h.listen("networkstatus-online",function(){if(a.j&&T("offline_error_handling")){var b=vs().get("errors",!0);if(b){for(var c in b)if(b[c]){var d=new U(c,"sent via offline_errors");d.name=b[c].name;d.stack=b[c].stack;d.level=b[c].level;Qm(d)}vs().set("errors",{},2592E3,!0)}}})}
v(zs,fi);function As(){if(!zs.instance){var a=F("yt.networkStatusManager.instance")||new zs;E("yt.networkStatusManager.instance",a);zs.instance=a}return zs.instance}
p=zs.prototype;p.ta=function(){return this.h.ta()};
p.mb=function(a){this.h.h=a};
p.xe=function(){var a=window.navigator.onLine;return a===void 0?!0:a};
p.ne=function(){this.j=!0};
p.listen=function(a,b){return this.h.listen(a,b)};
p.Fc=function(a){a=Uj(this.h,a);a.then(function(b){T("use_cfr_monitor")&&ys().requestComplete("generate_204",b)});
return a};
zs.prototype.sendNetworkCheckRequest=zs.prototype.Fc;zs.prototype.listen=zs.prototype.listen;zs.prototype.enableErrorFlushing=zs.prototype.ne;zs.prototype.getWindowStatus=zs.prototype.xe;zs.prototype.networkStatusHint=zs.prototype.mb;zs.prototype.isNetworkAvailable=zs.prototype.ta;zs.getInstance=As;function Bs(a){a=a===void 0?{}:a;fi.call(this);var b=this;this.h=this.u=0;this.j=As();var c=F("yt.networkStatusManager.instance.listen").bind(this.j);c&&(a.rateLimit?(this.rateLimit=a.rateLimit,c("networkstatus-online",function(){Cs(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){Cs(b,"publicytnetworkstatus-offline")})):(c("networkstatus-online",function(){gi(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){gi(b,"publicytnetworkstatus-offline")})))}
v(Bs,fi);Bs.prototype.ta=function(){var a=F("yt.networkStatusManager.instance.isNetworkAvailable");return a?a.bind(this.j)():!0};
Bs.prototype.mb=function(a){var b=F("yt.networkStatusManager.instance.networkStatusHint").bind(this.j);b&&b(a)};
Bs.prototype.Fc=function(a){var b=this,c;return B(function(d){c=F("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(b.j);return T("skip_network_check_if_cfr")&&ys().isEndpointCFR("generate_204")?d.return(new Promise(function(e){var f;b.mb(((f=window.navigator)==null?void 0:f.onLine)||!0);e(b.ta())})):c?d.return(c(a)):d.return(!0)})};
function Cs(a,b){a.rateLimit?a.h?(Xj.qa(a.u),a.u=Xj.pa(function(){a.o!==b&&(gi(a,b),a.o=b,a.h=V())},a.rateLimit-(V()-a.h))):(gi(a,b),a.o=b,a.h=V()):gi(a,b)}
;var Ds;function Es(){var a=Pr.call;Ds||(Ds=new Bs({Th:!0,Jh:!0}));a.call(Pr,this,{ga:{de:is,xb:hs,vd:ds,Ie:es,Zc:gs,set:bs},fa:Ds,handleError:function(b,c,d){var e,f=d==null?void 0:(e=d.error)==null?void 0:e.code;if(f===400||f===415){var g;b=new U(b.message,c,d==null?void 0:(g=d.error)==null?void 0:g.code);Rm(b,void 0,void 0,void 0,!0)}else Qm(b)},
zb:Rm,sendFn:Fs,now:V,Td:ws,Ca:Uo(),Yc:"publicytnetworkstatus-online",Vc:"publicytnetworkstatus-offline",oc:!0,lc:.1,Bc:ln("potential_esf_error_limit",10),W:T,Sb:!(mo()&&Gs())});this.j=new zj;T("networkless_immediately_drop_all_requests")&&js();Bq("LogsDatabaseV2")}
v(Es,Pr);function Hs(){var a=F("yt.networklessRequestController.instance");a||(a=new Es,E("yt.networklessRequestController.instance",a),T("networkless_logging")&&qq().then(function(b){a.V=b;Rr(a);a.j.resolve();a.oc&&Math.random()<=a.lc&&a.V&&ps(a.V);T("networkless_immediately_drop_sw_health_store")&&Is(a)}));
return a}
Es.prototype.writeThenSend=function(a,b){b||(b={});b=Js(a,b);mo()||(this.h=!1);Pr.prototype.writeThenSend.call(this,a,b)};
Es.prototype.sendThenWrite=function(a,b,c){b||(b={});b=Js(a,b);mo()||(this.h=!1);Pr.prototype.sendThenWrite.call(this,a,b,c)};
Es.prototype.sendAndWrite=function(a,b){b||(b={});b=Js(a,b);mo()||(this.h=!1);Pr.prototype.sendAndWrite.call(this,a,b)};
Es.prototype.awaitInitialization=function(){return this.j.promise};
function Is(a){var b;B(function(c){if(!a.V)throw b=vp("clearSWHealthLogsDb"),b;return c.return(qs(a.V).catch(function(d){a.handleError(d)}))})}
function Fs(a,b,c,d){d=d===void 0?!1:d;b=T("web_fp_via_jspb")?Object.assign({},b):b;T("use_cfr_monitor")&&Ks(a,b);if(T("use_request_time_ms_header"))b.headers&&bn(a)&&(b.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(V())));else{var e;if((e=b.postParams)==null?0:e.requestTimeMs)b.postParams.requestTimeMs=Math.round(V())}if(c&&Object.keys(b).length===0){var f=f===void 0?"":f;var g=g===void 0?!1:g;var h=h===void 0?!1:h;if(a)if(f)qn(a,void 0,"POST",f,void 0);else if(R("USE_NET_AJAX_FOR_PING_TRANSPORT",
!1)||h)qn(a,void 0,"GET","",void 0,void 0,g,h);else{b:{try{c:{var k=new db({url:a});if(ab(k.h,"dsh")==="1")var l=null;else{var m=ab(k.h,"ae");if(m==="1"){var n=ab(k.h,"adurl");if(n)try{l={version:3,ke:decodeURIComponent(n),ae:bb(k.h,"act=1","ri=1",eb(k))};break c}catch(Z){}}l=m==="2"?{version:4,ke:bb(k.h,"dct=1","suid="+k.i,""),ae:bb(k.h,"act=1","ri=1","suid="+k.i)}:null}}if(l){var r=lc(a),t;if(!(t=!r||!r.endsWith("/aclk"))){var w=a.search(tc),x=sc(a,0,"ri",w);if(x<0)var z=null;else{var G=a.indexOf("&",
x);if(G<0||G>w)G=w;z=fc(a.slice(x+3,G!==-1?G:0))}t=z!=="1"}var H=!t;break b}}catch(Z){}H=!1}if(H){b:{try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")){var S=!0;break b}}catch(Z){}S=!1}c=S?!0:!1}else c=!1;c||ts(a)}}else b.compress?b.postBody?(typeof b.postBody!=="string"&&(b.postBody=JSON.stringify(b.postBody)),Er(a,b.postBody,b,un,d)):Er(a,JSON.stringify(b.postParams),b,tn,d):un(a,b)}
function Js(a,b){T("use_event_time_ms_header")&&bn(a)&&(b.headers||(b.headers={}),b.headers["X-Goog-Event-Time"]=JSON.stringify(Math.round(V())));return b}
function Ks(a,b){var c=b.onError?b.onError:function(){};
b.onError=function(e,f){ys().requestComplete(a,!1);c(e,f)};
var d=b.onSuccess?b.onSuccess:function(){};
b.onSuccess=function(e,f){ys().requestComplete(a,!0);d(e,f)}}
function Gs(){return kc(document.location.toString())!=="www.youtube-nocookie.com"}
;var Ls=!1,Ms=D.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:Ls};E("ytNetworklessLoggingInitializationOptions",Ms);function Rs(){var a;B(function(b){if(b.h==1)return b.yield(qq(),2);a=b.i;if(!a||!mo()&&!T("nwl_init_require_datasync_id_killswitch")||!Gs())return b.B(0);Ls=!0;Ms.isNwlInitialized=Ls;return b.yield(Hs().awaitInitialization(),0)})}
;function Ss(a){var b=this;this.config_=null;a?this.config_=a:Sq()&&(this.config_=Tq());po(function(){Nr(b)},5E3)}
Ss.prototype.isReady=function(){!this.config_&&Sq()&&(this.config_=Tq());return!!this.config_};
function Or(a,b,c,d){function e(n){n=n===void 0?!1:n;var r;if(d.retry&&h!="www.youtube-nocookie.com"&&(n||T("skip_ls_gel_retry")||g.headers["Content-Type"]!=="application/json"||(r=Lr(b,c,l,k)),r)){var t=g.onSuccess,w=g.onFetchSuccess;g.onSuccess=function(G,H){Mr(r);t(G,H)};
c.onFetchSuccess=function(G,H){Mr(r);w(G,H)}}try{if(n&&d.retry&&!d.networklessOptions.bypassNetworkless)g.method="POST",d.networklessOptions.writeThenSend?Hs().writeThenSend(m,g):Hs().sendAndWrite(m,g);
else if(d.compress){var x=!d.networklessOptions.writeThenSend;if(g.postBody){var z=g.postBody;typeof z!=="string"&&(z=JSON.stringify(g.postBody));Er(m,z,g,un,x)}else Er(m,JSON.stringify(g.postParams),g,tn,x)}else T("web_all_payloads_via_jspb")?un(m,g):tn(m,g)}catch(G){if(G.name==="InvalidAccessError")r&&(Mr(r),r=0),Rm(Error("An extension is blocking network request."));else throw G;}r&&po(function(){Nr(a)},5E3)}
!R("VISITOR_DATA")&&b!=="visitor_id"&&Math.random()<.01&&Rm(new U("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new U("innertube xhrclient not ready",b,c,d);Qm(f);throw f;}var g={headers:d.headers||{},method:"POST",postParams:c,postBody:d.postBody,postBodyFormat:d.postBodyFormat||"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(n,r){if(d.onSuccess)d.onSuccess(r)},
onFetchSuccess:function(n){if(d.onSuccess)d.onSuccess(n)},
onError:function(n,r){if(d.onError)d.onError(r)},
onFetchError:function(n){if(d.onError)d.onError(n)},
timeout:d.timeout,withCredentials:!0,compress:d.compress};g.headers["Content-Type"]||(g.headers["Content-Type"]="application/json");var h="";(f=a.config_.De)&&(h=f);var k=a.config_.Ee||!1,l=Vq(k,h,d);Object.assign(g.headers,l);g.headers.Authorization&&!h&&k&&(g.headers["x-origin"]=window.location.origin);var m=$m(""+h+("/youtubei/"+a.config_.innertubeApiVersion+"/"+b),{alt:"json"});(F("ytNetworklessLoggingInitializationOptions")?Ms.isNwlInitialized:Ls)?oq().then(function(n){e(n)}):e(!1)}
;var Ts=0,Us=od?"webkit":nd?"moz":ld?"ms":kd?"o":"";E("ytDomDomGetNextId",F("ytDomDomGetNextId")||function(){return++Ts});var Vs={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function Ws(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.rotation=this.clientY=this.clientX=0;this.scale=1;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in Vs||(this[b]=a[b]);this.scale=a.scale;this.rotation=a.rotation;var c=a.target||a.srcElement;c&&c.nodeType==3&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;
if(d)try{d=d.nodeName?d:null}catch(e){d=null}else this.type=="mouseover"?d=a.fromElement:this.type=="mouseout"&&(d=a.toElement);this.relatedTarget=d;this.clientX=a.clientX!=void 0?a.clientX:a.pageX;this.clientY=a.clientY!=void 0?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||(this.type=="keypress"?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.h=a.pageX;this.i=a.pageY}}catch(e){}}
function Xs(a){if(document.body&&document.documentElement){var b=document.body.scrollTop+document.documentElement.scrollTop;a.h=a.clientX+(document.body.scrollLeft+document.documentElement.scrollLeft);a.i=a.clientY+b}}
Ws.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
Ws.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
Ws.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var xg=D.ytEventsEventsListeners||{};E("ytEventsEventsListeners",xg);var Ys=D.ytEventsEventsCounter||{count:0};E("ytEventsEventsCounter",Ys);
function Zs(a,b,c,d){d=d===void 0?{}:d;a.addEventListener&&(b!="mouseenter"||"onmouseenter"in document?b!="mouseleave"||"onmouseenter"in document?b=="mousewheel"&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return wg(function(e){var f=typeof e[4]==="boolean"&&e[4]==!!d,g=Oa(e[4])&&Oa(d)&&Bg(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
function $s(a,b,c,d){d=d===void 0?{}:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=Zs(a,b,c,d);if(e)return e;e=++Ys.count+"";var f=!(b!="mouseenter"&&b!="mouseleave"||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new Ws(h);if(!Lg(h.relatedTarget,function(k){return k==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new Ws(h);
h.currentTarget=a;return c.call(a,h)};
g=Pm(g);a.addEventListener?(b=="mouseenter"&&f?b="mouseover":b=="mouseleave"&&f?b="mouseout":b=="mousewheel"&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),at()||typeof d==="boolean"?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);xg[e]=[a,b,c,g,d];return e}
function bt(a){a&&(typeof a=="string"&&(a=[a]),Tb(a,function(b){if(b in xg){var c=xg[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?at()||typeof c==="boolean"?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete xg[b]}}))}
var at=vi(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});function ct(a){this.G=a;this.h=null;this.o=0;this.A=null;this.u=0;this.i=[];for(a=0;a<4;a++)this.i.push(0);this.j=0;this.U=$s(window,"mousemove",Ua(this.Y,this));a=Ua(this.P,this);typeof a==="function"&&(a=Pm(a));this.Z=window.setInterval(a,25)}
$a(ct,I);ct.prototype.Y=function(a){a.h===void 0&&Xs(a);var b=a.h;a.i===void 0&&Xs(a);this.h=new sg(b,a.i)};
ct.prototype.P=function(){if(this.h){var a=V();if(this.o!=0){var b=this.A,c=this.h,d=b.x-c.x;b=b.y-c.y;d=Math.sqrt(d*d+b*b)/(a-this.o);this.i[this.j]=Math.abs((d-this.u)/this.u)>.5?1:0;for(c=b=0;c<4;c++)b+=this.i[c]||0;b>=3&&this.G();this.u=d}this.o=a;this.A=this.h;this.j=(this.j+1)%4}};
ct.prototype.ba=function(){window.clearInterval(this.Z);bt(this.U)};var dt={};
function et(a){var b=a===void 0?{}:a;a=b.Ue===void 0?!1:b.Ue;b=b.oe===void 0?!0:b.oe;if(F("_lact",window)==null){var c=parseInt(R("LACT"),10);c=isFinite(c)?Date.now()-Math.max(c,0):-1;E("_lact",c,window);E("_fact",c,window);c==-1&&ft();$s(document,"keydown",ft);$s(document,"keyup",ft);$s(document,"mousedown",ft);$s(document,"mouseup",ft);a?$s(window,"touchmove",function(){gt("touchmove",200)},{passive:!0}):($s(window,"resize",function(){gt("resize",200)}),b&&$s(window,"scroll",function(){gt("scroll",200)}));
new ct(function(){gt("mouse",100)});
$s(document,"touchstart",ft,{passive:!0});$s(document,"touchend",ft,{passive:!0})}}
function gt(a,b){dt[a]||(dt[a]=!0,Xj.pa(function(){ft();dt[a]=!1},b))}
function ft(){F("_lact",window)==null&&et();var a=Date.now();E("_lact",a,window);F("_fact",window)==-1&&E("_fact",a,window);(a=F("ytglobal.ytUtilActivityCallback_"))&&a()}
function ht(){var a=F("_lact",window);return a==null?-1:Math.max(Date.now()-a,0)}
;var jt=D.ytPubsubPubsubInstance||new N,kt=D.ytPubsubPubsubSubscribedKeys||{},lt=D.ytPubsubPubsubTopicToKeys||{},mt=D.ytPubsubPubsubIsSynchronous||{};function nt(a,b){var c=ot();if(c&&b){var d=c.subscribe(a,function(){function e(){kt[d]&&b.apply&&typeof b.apply=="function"&&b.apply(window,f)}
var f=arguments;try{mt[a]?e():hn(e,0)}catch(g){Qm(g)}},void 0);
kt[d]=!0;lt[a]||(lt[a]=[]);lt[a].push(d);return d}return 0}
function pt(a){var b=ot();b&&(typeof a==="number"?a=[a]:typeof a==="string"&&(a=[parseInt(a,10)]),Tb(a,function(c){b.unsubscribeByKey(c);delete kt[c]}))}
function qt(a,b){var c=ot();c&&c.publish.apply(c,arguments)}
function rt(a){var b=ot();if(b)if(b.clear(a),a)st(a);else for(var c in lt)st(c)}
function ot(){return D.ytPubsubPubsubInstance}
function st(a){lt[a]&&(a=lt[a],Tb(a,function(b){kt[b]&&delete kt[b]}),a.length=0)}
N.prototype.subscribe=N.prototype.subscribe;N.prototype.unsubscribeByKey=N.prototype.fc;N.prototype.publish=N.prototype.sb;N.prototype.clear=N.prototype.clear;E("ytPubsubPubsubInstance",jt);E("ytPubsubPubsubTopicToKeys",lt);E("ytPubsubPubsubIsSynchronous",mt);E("ytPubsubPubsubSubscribedKeys",kt);var tt=Symbol("injectionDeps");function ut(a){this.name=a}
ut.prototype.toString=function(){return"InjectionToken("+this.name+")"};
function vt(a){this.key=a}
function wt(a){return new vt(a)}
function xt(){this.i=new Map;this.j=new Map;this.h=new Map}
function zt(a,b){a.i.set(b.pb,b);var c=a.j.get(b.pb);if(c)try{c.ci(a.resolve(b.pb))}catch(d){c.ai(d)}}
xt.prototype.resolve=function(a){return a instanceof vt?At(this,a.key,[],!0):At(this,a,[])};
function At(a,b,c,d){d=d===void 0?!1:d;if(c.indexOf(b)>-1)throw Error("Deps cycle for: "+b);if(a.h.has(b))return a.h.get(b);if(!a.i.has(b)){if(d)return;throw Error("No provider for: "+b);}d=a.i.get(b);c.push(b);if(d.kd!==void 0)var e=d.kd;else if(d.Bf)e=d[tt]?Bt(a,d[tt],c):[],e=d.Bf.apply(d,A(e));else if(d.Hc){e=d.Hc;var f=e[tt]?Bt(a,e[tt],c):[];e=new (Function.prototype.bind.apply(e,[null].concat(A(f))))}else throw Error("Could not resolve providers for: "+b);c.pop();d.ji||a.h.set(b,e);return e}
function Bt(a,b,c){return b?b.map(function(d){return d instanceof vt?At(a,d.key,c,!0):At(a,d,c)}):[]}
;var Ct;function Dt(){Ct||(Ct=new xt);return Ct}
;var Et=window;function Ft(){var a,b;return"h5vcc"in Et&&((a=Et.h5vcc.traceEvent)==null?0:a.traceBegin)&&((b=Et.h5vcc.traceEvent)==null?0:b.traceEnd)?1:"performance"in Et&&Et.performance.mark&&Et.performance.measure?2:0}
function Gt(a){var b=Ft();switch(b){case 1:Et.h5vcc.traceEvent.traceBegin("YTLR",a);break;case 2:Et.performance.mark(a+"-start");break;case 0:break;default:Eb(b,"unknown trace type")}}
function Ht(a){var b=Ft();switch(b){case 1:Et.h5vcc.traceEvent.traceEnd("YTLR",a);break;case 2:b=a+"-start";var c=a+"-end";Et.performance.mark(c);Et.performance.measure(a,b,c);break;case 0:break;default:Eb(b,"unknown trace type")}}
;var It=T("web_enable_lifecycle_monitoring")&&Ft()!==0,Jt=T("web_enable_lifecycle_monitoring");function Kt(a){var b,c;(c=(b=window).onerror)==null||c.call(b,a.message,"",0,0,a)}
;function Lt(a){var b=this;var c=c===void 0?0:c;var d=d===void 0?Uo():d;this.j=c;this.scheduler=d;this.i=new zj;this.h=a;for(a={jb:0};a.jb<this.h.length;a={Ac:void 0,jb:a.jb},a.jb++)a.Ac=this.h[a.jb],c=function(e){return function(){e.Ac.Pc();b.h[e.jb].Cc=!0;b.h.every(function(f){return f.Cc===!0})&&b.i.resolve()}}(a),d=this.getPriority(a.Ac),d=this.scheduler.Sa(c,d),this.h[a.jb]=Object.assign({},a.Ac,{Pc:c,
jobId:d})}
function Mt(a){var b=Array.from(a.h.keys()).sort(function(d,e){return a.getPriority(a.h[e])-a.getPriority(a.h[d])});
b=y(b);for(var c=b.next();!c.done;c=b.next())c=a.h[c.value],c.jobId===void 0||c.Cc||(a.scheduler.qa(c.jobId),a.scheduler.Sa(c.Pc,10))}
Lt.prototype.cancel=function(){for(var a=y(this.h),b=a.next();!b.done;b=a.next())b=b.value,b.jobId===void 0||b.Cc||this.scheduler.qa(b.jobId),b.Cc=!0;this.i.resolve()};
Lt.prototype.getPriority=function(a){var b;return(b=a.priority)!=null?b:this.j};function Nt(a){this.state=a;this.plugins=[];this.o=void 0;this.A={};It&&Gt(this.state)}
p=Nt.prototype;p.install=function(a){this.plugins.push(a);return this};
p.uninstall=function(){var a=this;C.apply(0,arguments).forEach(function(b){b=a.plugins.indexOf(b);b>-1&&a.plugins.splice(b,1)})};
p.transition=function(a,b){var c=this;It&&Ht(this.state);var d=this.transitions.find(function(f){return Array.isArray(f.from)?f.from.find(function(g){return g===c.state&&f.to===a}):f.from===c.state&&f.to===a});
if(d){this.j&&(Mt(this.j),this.j=void 0);Ot(this,a,b);this.state=a;It&&Gt(this.state);d=d.action.bind(this);var e=this.plugins.filter(function(f){return f[a]}).map(function(f){return f[a]});
d(Pt(this,e),b)}else throw Error("no transition specified from "+this.state+" to "+a);};
function Pt(a,b){var c=b.filter(function(e){return Qt(a,e)===10}),d=b.filter(function(e){return Qt(a,e)!==10});
return a.A.ii?function(){var e=C.apply(0,arguments);return B(function(f){if(f.h==1)return f.yield(a.af.apply(a,[c].concat(A(e))),2);a.Ld.apply(a,[d].concat(A(e)));f.h=0})}:function(){var e=C.apply(0,arguments);
a.bf.apply(a,[c].concat(A(e)));a.Ld.apply(a,[d].concat(A(e)))}}
p.bf=function(a){for(var b=C.apply(1,arguments),c=Uo(),d=y(a),e=d.next(),f={};!e.done;f={Ub:void 0},e=d.next())f.Ub=e.value,c.Mb(function(g){return function(){Rt(g.Ub.name);St(function(){return g.Ub.callback.apply(g.Ub,A(b))});
Tt(g.Ub.name)}}(f))};
p.af=function(a){var b=C.apply(1,arguments),c,d,e,f,g;return B(function(h){h.h==1&&(c=Uo(),d=y(a),e=d.next(),f={});if(h.h!=3){if(e.done)return h.B(0);f.Ya=e.value;f.hc=void 0;g=function(k){return function(){Rt(k.Ya.name);var l=St(function(){return k.Ya.callback.apply(k.Ya,A(b))});
ne(l)?k.hc=T("web_lifecycle_error_handling_killswitch")?l.then(function(){Tt(k.Ya.name)}):l.then(function(){Tt(k.Ya.name)},function(m){Kt(m);
Tt(k.Ya.name)}):Tt(k.Ya.name)}}(f);
c.Mb(g);return f.hc?h.yield(f.hc,3):h.B(3)}f={Ya:void 0,hc:void 0};e=d.next();return h.B(2)})};
p.Ld=function(a){var b=C.apply(1,arguments),c=this,d=a.map(function(e){return{Pc:function(){Rt(e.name);St(function(){return e.callback.apply(e,A(b))});
Tt(e.name)},
priority:Qt(c,e)}});
d.length&&(this.j=new Lt(d))};
function Qt(a,b){var c,d;return(d=(c=a.o)!=null?c:b.priority)!=null?d:0}
function Rt(a){It&&a&&Gt(a)}
function Tt(a){It&&a&&Ht(a)}
function Ot(a,b,c){Jt&&console.groupCollapsed&&console.groupEnd&&(console.groupCollapsed("["+a.constructor.name+"] '"+a.state+"' to '"+b+"'"),console.log("with message: ",c),console.groupEnd())}
da.Object.defineProperties(Nt.prototype,{currentState:{configurable:!0,enumerable:!0,get:function(){return this.state}}});
function St(a){if(T("web_lifecycle_error_handling_killswitch"))return a();try{return a()}catch(b){Kt(b)}}
;function Ut(a){Nt.call(this,a===void 0?"none":a);this.h=null;this.o=10;this.transitions=[{from:"none",to:"application_navigating",action:this.i},{from:"application_navigating",to:"none",action:this.u},{from:"application_navigating",to:"application_navigating",action:function(){}},
{from:"none",to:"none",action:function(){}}]}
var Vt;v(Ut,Nt);Ut.prototype.i=function(a,b){var c=this;this.h=po(function(){c.currentState==="application_navigating"&&c.transition("none")},5E3);
a(b==null?void 0:b.event)};
Ut.prototype.u=function(a,b){this.h&&(Xj.qa(this.h),this.h=null);a(b==null?void 0:b.event)};
function Wt(){Vt||(Vt=new Ut);return Vt}
;var Xt=[];E("yt.logging.transport.getScrapedGelPayloads",function(){return Xt});function Yt(){this.store={};this.h={}}
Yt.prototype.storePayload=function(a,b){a=Zt(a);this.store[a]?this.store[a].push(b):(this.h={},this.store[a]=[b]);T("more_accurate_gel_parser")&&(b=new CustomEvent("TRANSPORTING_NEW_EVENT"),window.dispatchEvent(b));return a};
Yt.prototype.smartExtractMatchingEntries=function(a){if(!a.keys.length)return[];for(var b=$t(this,a.keys.splice(0,1)[0]),c=[],d=0;d<b.length;d++)this.store[b[d]]&&a.sizeLimit&&(this.store[b[d]].length<=a.sizeLimit?(c.push.apply(c,A(this.store[b[d]])),delete this.store[b[d]]):c.push.apply(c,A(this.store[b[d]].splice(0,a.sizeLimit))));(a==null?0:a.sizeLimit)&&c.length<(a==null?void 0:a.sizeLimit)&&(a.sizeLimit-=c.length,c.push.apply(c,A(this.smartExtractMatchingEntries(a))));return c};
Yt.prototype.extractMatchingEntries=function(a){a=$t(this,a);for(var b=[],c=0;c<a.length;c++)this.store[a[c]]&&(b.push.apply(b,A(this.store[a[c]])),delete this.store[a[c]]);return b};
Yt.prototype.getSequenceCount=function(a){a=$t(this,a);for(var b=0,c=0;c<a.length;c++){var d=void 0;b+=((d=this.store[a[c]])==null?void 0:d.length)||0}return b};
function $t(a,b){var c=Zt(b);if(a.h[c])return a.h[c];var d=Object.keys(a.store)||[];if(d.length<=1&&Zt(b)===d[0])return d;for(var e=[],f=0;f<d.length;f++){var g=d[f].split("/");if(au(b.auth,g[0])){var h=b.isJspb;au(h===void 0?"undefined":h?"true":"false",g[1])&&au(b.cttAuthInfo,g[2])&&(h=b.tier,h=h===void 0?"undefined":JSON.stringify(h),au(h,g[3])&&e.push(d[f]))}}return a.h[c]=e}
function au(a,b){return a===void 0||a==="undefined"?!0:a===b}
Yt.prototype.getSequenceCount=Yt.prototype.getSequenceCount;Yt.prototype.extractMatchingEntries=Yt.prototype.extractMatchingEntries;Yt.prototype.smartExtractMatchingEntries=Yt.prototype.smartExtractMatchingEntries;Yt.prototype.storePayload=Yt.prototype.storePayload;function Zt(a){return[a.auth===void 0?"undefined":a.auth,a.isJspb===void 0?"undefined":a.isJspb,a.cttAuthInfo===void 0?"undefined":a.cttAuthInfo,a.tier===void 0?"undefined":a.tier].join("/")}
;function bu(a,b){if(a)return a[b.name]}
;var cu=ln("initial_gel_batch_timeout",2E3),du=ln("gel_queue_timeout_max_ms",6E4),eu=ln("gel_min_batch_size",5),fu=void 0;function gu(){this.o=this.h=this.i=0;this.j=!1}
var hu=new gu,iu=new gu,ju=new gu,ku=new gu,lu,mu=!0,nu=D.ytLoggingTransportTokensToCttTargetIds_||{};E("ytLoggingTransportTokensToCttTargetIds_",nu);var ou={};function pu(){var a=F("yt.logging.ims");a||(a=new Yt,E("yt.logging.ims",a));return a}
function qu(a,b){if(a.endpoint==="log_event"){ru(a);var c=su(a),d=tu(a.payload)||"";a:{if(T("enable_web_tiered_gel")){var e=ks[d||""];var f,g,h,k=Dt().resolve(wt(Nq))==null?void 0:(f=Oq())==null?void 0:(g=f.loggingHotConfig)==null?void 0:(h=g.eventLoggingConfig)==null?void 0:h.payloadPolicies;if(k)for(f=0;f<k.length;f++)if(k[f].payloadNumber===e){e=k[f];break a}}e=void 0}k=200;if(e){if(e.enabled===!1&&!T("web_payload_policy_disabled_killswitch"))return;k=uu(e.tier);if(k===400){vu(a,b);return}}ou[c]=
!0;c={cttAuthInfo:c,isJspb:!1,tier:k};pu().storePayload(c,a.payload);wu(b,c,d==="gelDebuggingEvent")}}
function wu(a,b,c){function d(){xu({writeThenSend:!0},void 0,e,b.tier)}
var e=!1;e=e===void 0?!1:e;c=c===void 0?!1:c;a&&(fu=new a);a=ln("tvhtml5_logging_max_batch_ads_fork")||ln("tvhtml5_logging_max_batch")||ln("web_logging_max_batch")||100;var f=V(),g=yu(e,b.tier),h=g.o;c&&(g.j=!0);c=0;b&&(c=pu().getSequenceCount(b));c>=1E3?d():c>=a?lu||(lu=zu(function(){d();lu=void 0},0)):f-h>=10&&(Au(e,b.tier),g.o=f)}
function vu(a,b){if(a.endpoint==="log_event"){T("more_accurate_gel_parser")&&pu().storePayload({isJspb:!1},a.payload);ru(a);var c=su(a),d=new Map;d.set(c,[a.payload]);var e=tu(a.payload)||"";b&&(fu=new b);return new wi(function(f,g){fu&&fu.isReady()?Bu(d,fu,f,g,{bypassNetworkless:!0},!0,e==="gelDebuggingEvent"):f()})}}
function su(a){var b="";if(a.dangerousLogToVisitorSession)b="visitorOnlyApprovedKey";else if(a.cttAuthInfo){b=a.cttAuthInfo;var c={};b.videoId?c.videoId=b.videoId:b.playlistId&&(c.playlistId=b.playlistId);nu[a.cttAuthInfo.token]=c;b=a.cttAuthInfo.token}return b}
function xu(a,b,c,d){a=a===void 0?{}:a;c=c===void 0?!1:c;new wi(function(e,f){var g=yu(c,d),h=g.j;g.j=!1;Cu(g.i);Cu(g.h);g.h=0;fu&&fu.isReady()?d===void 0&&T("enable_web_tiered_gel")?Du(e,f,a,b,c,300,h):Du(e,f,a,b,c,d,h):(Au(c,d),e())})}
function Du(a,b,c,d,e,f,g){var h=fu;c=c===void 0?{}:c;e=e===void 0?!1:e;f=f===void 0?200:f;g=g===void 0?!1:g;var k=new Map,l={isJspb:e,cttAuthInfo:d,tier:f};e={isJspb:e,cttAuthInfo:d};if(d!==void 0)f=T("enable_web_tiered_gel")?pu().smartExtractMatchingEntries({keys:[l,e],sizeLimit:1E3}):pu().extractMatchingEntries(e),k.set(d,f);else for(d=y(Object.keys(ou)),l=d.next();!l.done;l=d.next())l=l.value,e=T("enable_web_tiered_gel")?pu().smartExtractMatchingEntries({keys:[{isJspb:!1,cttAuthInfo:l,tier:f},
{isJspb:!1,cttAuthInfo:l}],sizeLimit:1E3}):pu().extractMatchingEntries({isJspb:!1,cttAuthInfo:l}),e.length>0&&k.set(l,e),(T("web_fp_via_jspb_and_json")&&c.writeThenSend||!T("web_fp_via_jspb_and_json"))&&delete ou[l];Bu(k,h,a,b,c,!1,g)}
function Au(a,b){function c(){xu({writeThenSend:!0},void 0,a,b)}
a=a===void 0?!1:a;b=b===void 0?200:b;var d=yu(a,b),e=d===ku||d===ju?5E3:du;T("web_gel_timeout_cap")&&!d.h&&(e=zu(function(){c()},e),d.h=e);
Cu(d.i);e=R("LOGGING_BATCH_TIMEOUT",ln("web_gel_debounce_ms",1E4));T("shorten_initial_gel_batch_timeout")&&mu&&(e=cu);e=zu(function(){ln("gel_min_batch_size")>0?pu().getSequenceCount({cttAuthInfo:void 0,isJspb:a,tier:b})>=eu&&c():c()},e);
d.i=e}
function Bu(a,b,c,d,e,f,g){e=e===void 0?{}:e;var h=Math.round(V()),k=a.size,l=(g===void 0?0:g)&&T("vss_through_gel_video_stats")?"video_stats":"log_event";a=y(a);var m=a.next();for(g={};!m.done;g={Uc:void 0,batchRequest:void 0,dangerousLogToVisitorSession:void 0,Xc:void 0,Wc:void 0},m=a.next()){var n=y(m.value);m=n.next().value;n=n.next().value;g.batchRequest=Dg({context:Uq(b.config_||Tq())});if(!Ma(n)&&!T("throw_err_when_logevent_malformed_killswitch")){d();break}g.batchRequest.events=n;(n=nu[m])&&
Eu(g.batchRequest,m,n);delete nu[m];g.dangerousLogToVisitorSession=m==="visitorOnlyApprovedKey";Fu(g.batchRequest,h,g.dangerousLogToVisitorSession);T("always_send_and_write")&&(e.writeThenSend=!1);g.Xc=function(r){T("start_client_gcf")&&Xj.pa(function(){return B(function(t){return t.yield(Gu(r),0)})});
k--;k||c()};
g.Uc=0;g.Wc=function(r){return function(){r.Uc++;if(e.bypassNetworkless&&r.Uc===1)try{Or(b,l,r.batchRequest,Hu({writeThenSend:!0},r.dangerousLogToVisitorSession,r.Xc,r.Wc,f)),mu=!1}catch(t){Qm(t),d()}k--;k||c()}}(g);
try{Or(b,l,g.batchRequest,Hu(e,g.dangerousLogToVisitorSession,g.Xc,g.Wc,f)),mu=!1}catch(r){Qm(r),d()}}}
function Hu(a,b,c,d,e){a={retry:!0,onSuccess:c,onError:d,networklessOptions:a,dangerousLogToVisitorSession:b,vh:!!e,headers:{},postBodyFormat:"",postBody:"",compress:T("compress_gel")||T("compress_gel_lr")};Iu()&&(a.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(V())));return a}
function Fu(a,b,c){Iu()||(a.requestTimeMs=String(b));T("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=R("EVENT_ID"))&&((c=R("BATCH_CLIENT_COUNTER")||0)||(c=Math.floor(Math.random()*65535/2)),c++,c>65535&&(c=1),Lm("BATCH_CLIENT_COUNTER",c),a.serializedClientEventId={serializedEventId:b,clientCounter:String(c)})}
function Eu(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function ru(a){var b=kn("il_payload_scraping");b=(b!==void 0?String(b):"")==="enable_il_payload_scraping";if(!F("yt.logging.transport.enableScrapingForTest"))if(b)Xt=[],E("yt.logging.transport.enableScrapingForTest",!0),E("yt.logging.transport.scrapedPayloadsForTesting",Xt),E("yt.logging.transport.payloadToScrape","visualElementShown visualElementHidden visualElementAttached screenCreated visualElementGestured visualElementStateChanged".split(" ")),E("yt.logging.transport.getScrapedPayloadFromClientEventsFunction"),
E("yt.logging.transport.scrapeClientEvent",!0);else return;b=F("yt.logging.transport.scrapedPayloadsForTesting");var c=F("yt.logging.transport.payloadToScrape"),d=F("yt.logging.transport.scrapeClientEvent");if(c&&c.length>=1)for(var e=0;e<c.length;e++)if(a&&a.payload[c[e]])if(d)b.push(a.payload);else{var f=void 0;b.push(((f=a)==null?void 0:f.payload)[c[e]])}E("yt.logging.transport.scrapedPayloadsForTesting",b)}
function Iu(){return T("use_request_time_ms_header")||T("lr_use_request_time_ms_header")}
function zu(a,b){return T("transport_use_scheduler")===!1?hn(a,b):T("logging_avoid_blocking_during_navigation")||T("lr_logging_avoid_blocking_during_navigation")?po(function(){if(Wt().currentState==="none")a();else{var c={};Wt().install((c.none={callback:a},c))}},b):po(a,b)}
function Cu(a){T("transport_use_scheduler")?Xj.qa(a):window.clearTimeout(a)}
function Gu(a){var b,c,d,e,f,g,h,k,l,m;return B(function(n){return n.h==1?(d=(b=a)==null?void 0:(c=b.responseContext)==null?void 0:c.globalConfigGroup,e=bu(d,lm),g=(f=d)==null?void 0:f.hotHashData,h=bu(d,km),l=(k=d)==null?void 0:k.coldHashData,(m=Dt().resolve(wt(Nq)))?g?e?n.yield(Pq(m,g,e),2):n.yield(Pq(m,g),2):n.B(2):n.return()):l?h?n.yield(Qq(m,l,h),0):n.yield(Qq(m,l),0):n.B(0)})}
function yu(a,b){b=b===void 0?200:b;return a?b===300?ku:iu:b===300?ju:hu}
function tu(a){a=Object.keys(a);a=y(a);for(var b=a.next();!b.done;b=a.next())if(b=b.value,ks[b])return b}
function uu(a){switch(a){case "DELAYED_EVENT_TIER_UNSPECIFIED":return 0;case "DELAYED_EVENT_TIER_DEFAULT":return 100;case "DELAYED_EVENT_TIER_DISPATCH_TO_EMPTY":return 200;case "DELAYED_EVENT_TIER_FAST":return 300;case "DELAYED_EVENT_TIER_IMMEDIATE":return 400;default:return 200}}
;var Ju=D.ytLoggingGelSequenceIdObj_||{};E("ytLoggingGelSequenceIdObj_",Ju);
function Ku(a,b,c,d){d=d===void 0?{}:d;var e={},f=Math.round(d.timestamp||V());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;a=ht();e.context={lastActivityMs:String(d.timestamp||!isFinite(a)?-1:a)};d.sequenceGroup&&!T("web_gel_sequence_info_killswitch")&&(a=e.context,b=d.sequenceGroup,Ju[b]=b in Ju?Ju[b]+1:0,a.sequence={index:Ju[b],groupKey:b},d.endOfSequence&&delete Ju[d.sequenceGroup]);(d.sendIsolatedPayload?vu:qu)({endpoint:"log_event",payload:e,cttAuthInfo:d.cttAuthInfo,dangerousLogToVisitorSession:d.dangerousLogToVisitorSession},
c)}
;function dp(a,b,c){c=c===void 0?{}:c;var d=Ss;R("ytLoggingEventsDefaultDisabled",!1)&&Ss===Ss&&(d=null);Ku(a,b,d,c)}
;var Lu=new Set,Mu=0,Nu=0,Ou=0,Pu=[],Qu=[],Ru=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];function cp(a){Su(a)}
function Tu(a){Su(a,"WARNING")}
function Uu(a){a instanceof Error?Su(a):(a=Oa(a)?JSON.stringify(a):String(a),a=new U(a),a.name="RejectedPromiseError",Tu(a))}
function Su(a,b,c,d,e,f,g,h){f=f===void 0?{}:f;f.name=c||R("INNERTUBE_CONTEXT_CLIENT_NAME",1);f.version=d||R("INNERTUBE_CONTEXT_CLIENT_VERSION");c=f;b=b===void 0?"ERROR":b;g=g===void 0?!1:g;b=b===void 0?"ERROR":b;g=g===void 0?!1:g;if(a&&(a.hasOwnProperty("level")&&a.level&&(b=a.level),T("console_log_js_exceptions")&&(d=[],d.push("Name: "+a.name),d.push("Message: "+a.message),a.hasOwnProperty("params")&&d.push("Error Params: "+JSON.stringify(a.params)),a.hasOwnProperty("args")&&d.push("Error args: "+
JSON.stringify(a.args)),d.push("File name: "+a.fileName),d.push("Stacktrace: "+a.stack),d=d.join("\n"),window.console.log(d,a)),!(Mu>=5))){d=[];e=y(Qu);for(f=e.next();!f.done;f=e.next()){f=f.value;try{f()&&d.push(f())}catch(z){}}d=[].concat(A(Pu),A(d));var k=bc(a);e=k.message||"Unknown Error";f=k.name||"UnknownError";var l=k.stack||a.i||"Not available";if(l.startsWith(f+": "+e)){var m=l.split("\n");m.shift();l=m.join("\n")}m=k.lineNumber||"Not available";k=k.fileName||"Not available";var n=0;if(a.hasOwnProperty("args")&&
a.args&&a.args.length)for(var r=0;r<a.args.length&&!(n=Mn(a.args[r],"params."+r,c,n),n>=500);r++);else if(a.hasOwnProperty("params")&&a.params){var t=a.params;if(typeof a.params==="object")for(r in t){if(t[r]){var w="params."+r,x=On(t[r]);c[w]=x;n+=w.length+x.length;if(n>500)break}}else c.params=On(t)}if(d.length)for(r=0;r<d.length&&!(n=Mn(d[r],"params.context."+r,c,n),n>=500);r++);navigator.vendor&&!c.hasOwnProperty("vendor")&&(c["device.vendor"]=navigator.vendor);r={message:e,name:f,lineNumber:m,
fileName:k,stack:l,params:c,sampleWeight:1};c=Number(a.columnNumber);isNaN(c)||(r.lineNumber=r.lineNumber+":"+c);if(a.level==="IGNORED")a=0;else a:{a=In();c=y(a.Za);for(d=c.next();!d.done;d=c.next())if(d=d.value,r.message&&r.message.match(d.Vh)){a=d.weight;break a}a=y(a.Ua);for(c=a.next();!c.done;c=a.next())if(c=c.value,c.callback(r)){a=c.weight;break a}a=1}r.sampleWeight=a;a=y(Dn);for(c=a.next();!c.done;c=a.next())if(c=c.value,c.zc[r.name])for(e=y(c.zc[r.name]),d=e.next();!d.done;d=e.next())if(f=
d.value,d=r.message.match(f.regexp)){r.params["params.error.original"]=d[0];e=f.groups;f={};for(m=0;m<e.length;m++)f[e[m]]=d[m+1],r.params["params.error."+e[m]]=d[m+1];r.message=c.Tc(f);break}r.params||(r.params={});a=In();r.params["params.errorServiceSignature"]="msg="+a.Za.length+"&cb="+a.Ua.length;r.params["params.serviceWorker"]="false";D.document&&D.document.querySelectorAll&&(r.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));(new Gg(Hg,"sample")).constructor!==
Gg&&(r.params["params.fconst"]="true");window.yterr&&typeof window.yterr==="function"&&window.yterr(r);if(r.sampleWeight!==0&&!Lu.has(r.message)){if(g&&T("web_enable_error_204"))Vu(b===void 0?"ERROR":b,r);else{b=b===void 0?"ERROR":b;b==="ERROR"?(Jn.sb("handleError",r),T("record_app_crashed_web")&&Ou===0&&r.sampleWeight===1&&(Ou++,g={appCrashType:"APP_CRASH_TYPE_BREAKPAD"},T("report_client_error_with_app_crash_ks")||(g.systemHealth={crashData:{clientError:{logMessage:{message:r.message}}}}),dp("appCrashed",
g)),Nu++):b==="WARNING"&&Jn.sb("handleWarning",r);if(T("kevlar_gel_error_routing")){g=b;h=h===void 0?{}:h;b:{a=y(Ru);for(c=a.next();!c.done;c=a.next())if(jp(c.value.toLowerCase())){a=!0;break b}a=!1}if(a)h=void 0;else{c={stackTrace:r.stack};r.fileName&&(c.filename=r.fileName);a=r.lineNumber&&r.lineNumber.split?r.lineNumber.split(":"):[];a.length!==0&&(a.length!==1||isNaN(Number(a[0]))?a.length!==2||isNaN(Number(a[0]))||isNaN(Number(a[1]))||(c.lineNumber=Number(a[0]),c.columnNumber=Number(a[1])):c.lineNumber=
Number(a[0]));a={level:"ERROR_LEVEL_UNKNOWN",message:r.message,errorClassName:r.name,sampleWeight:r.sampleWeight};g==="ERROR"?a.level="ERROR_LEVEL_ERROR":g==="WARNING"&&(a.level="ERROR_LEVEL_WARNNING");c={isObfuscated:!0,browserStackInfo:c};h.pageUrl=window.location.href;h.kvPairs=[];R("FEXP_EXPERIMENTS")&&(h.experimentIds=R("FEXP_EXPERIMENTS"));d=R("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(!Mm("web_disable_gel_stp_ecatcher_killswitch")&&d)for(e=y(Object.keys(d)),f=e.next();!f.done;f=e.next())f=
f.value,h.kvPairs.push({key:f,value:String(d[f])});if(d=r.params)for(e=y(Object.keys(d)),f=e.next();!f.done;f=e.next())f=f.value,h.kvPairs.push({key:"client."+f,value:String(d[f])});d=R("SERVER_NAME");e=R("SERVER_VERSION");d&&e&&(h.kvPairs.push({key:"server.name",value:d}),h.kvPairs.push({key:"server.version",value:e}));h={errorMetadata:h,stackTrace:c,logMessage:a}}h&&(dp("clientError",h),(g==="ERROR"||T("errors_flush_gel_always_killswitch"))&&xu(void 0,void 0,!1))}T("suppress_error_204_logging")||
Vu(b,r)}try{Lu.add(r.message)}catch(z){}Mu++}}}
function Vu(a,b){var c=b.params||{};a={urlParams:{a:"logerror",t:"jserror",type:b.name,msg:b.message.substr(0,250),line:b.lineNumber,level:a,"client.name":c.name},postParams:{url:R("PAGE_NAME",window.location.href),file:b.fileName},method:"POST"};c.version&&(a["client.version"]=c.version);if(a.postParams){b.stack&&(a.postParams.stack=b.stack);b=y(Object.keys(c));for(var d=b.next();!d.done;d=b.next())d=d.value,a.postParams["client."+d]=c[d];if(c=R("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS"))for(b=y(Object.keys(c)),
d=b.next();!d.done;d=b.next())d=d.value,a.postParams[d]=c[d];(c=R("LAVA_VERSION"))&&(a.postParams["lava.version"]=c);c=R("SERVER_NAME");b=R("SERVER_VERSION");c&&b&&(a.postParams["server.name"]=c,a.postParams["server.version"]=b)}un(R("ECATCHER_REPORT_HOST","")+"/error_204",a)}
function Wu(a){var b=C.apply(1,arguments);a.args||(a.args=[]);Array.isArray(a.args)&&a.args.push.apply(a.args,A(b))}
;function Xu(){this.register=new Map}
function Yu(a){a=y(a.register.values());for(var b=a.next();!b.done;b=a.next())b.value.Zh("ABORTED")}
Xu.prototype.clear=function(){Yu(this);this.register.clear()};
var Zu=new Xu;var $u=Date.now().toString();
function av(){a:{if(window.crypto&&window.crypto.getRandomValues)try{var a=Array(16),b=new Uint8Array(16);window.crypto.getRandomValues(b);for(var c=0;c<a.length;c++)a[c]=b[c];var d=a;break a}catch(e){}d=Array(16);for(a=0;a<16;a++){b=Date.now();for(c=0;c<b%23;c++)d[a]=Math.random();d[a]=Math.floor(Math.random()*256)}if($u)for(a=1,b=0;b<$u.length;b++)d[a%16]^=d[(a-1)%16]/4^$u.charCodeAt(b),a++}a=[];for(b=0;b<d.length;b++)a.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(d[b]&63));
return a.join("")}
;var bv,cv=D.ytLoggingDocDocumentNonce_;cv||(cv=av(),E("ytLoggingDocDocumentNonce_",cv));bv=cv;function dv(a){this.h=a}
p=dv.prototype;p.getAsJson=function(){var a={};this.h.trackingParams!==void 0?a.trackingParams=this.h.trackingParams:(a.veType=this.h.veType,this.h.veCounter!==void 0&&(a.veCounter=this.h.veCounter),this.h.elementIndex!==void 0&&(a.elementIndex=this.h.elementIndex));this.h.dataElement!==void 0&&(a.dataElement=this.h.dataElement.getAsJson());this.h.youtubeData!==void 0&&(a.youtubeData=this.h.youtubeData);this.h.isCounterfactual&&(a.isCounterfactual=!0);return a};
p.getAsJspb=function(){var a=new nm;this.h.trackingParams!==void 0?a.setTrackingParams(this.h.trackingParams):(this.h.veType!==void 0&&xf(a,2,Oe(this.h.veType)),this.h.veCounter!==void 0&&xf(a,6,Oe(this.h.veCounter)),this.h.elementIndex!==void 0&&xf(a,3,Oe(this.h.elementIndex)),this.h.isCounterfactual&&xf(a,5,Ke(!0)));if(this.h.dataElement!==void 0){var b=this.h.dataElement.getAsJspb();Hf(a,nm,7,b)}this.h.youtubeData!==void 0&&Hf(a,mm,8,this.h.jspbYoutubeData);return a};
p.toString=function(){return JSON.stringify(this.getAsJson())};
p.isClientVe=function(){return!this.h.trackingParams&&!!this.h.veType};
p.getLoggingDirectives=function(){return this.h.loggingDirectives};function ev(a){return R("client-screen-nonce-store",{})[a===void 0?0:a]}
function fv(a,b){b=b===void 0?0:b;var c=R("client-screen-nonce-store");c||(c={},Lm("client-screen-nonce-store",c));c[b]=a}
function gv(a){a=a===void 0?0:a;return a===0?"ROOT_VE_TYPE":"ROOT_VE_TYPE."+a}
function hv(a){return R(gv(a===void 0?0:a))}
E("yt_logging_screen.getRootVeType",hv);function iv(){var a=R("csn-to-ctt-auth-info");a||(a={},Lm("csn-to-ctt-auth-info",a));return a}
function jv(){return Object.values(R("client-screen-nonce-store",{})).filter(function(a){return a!==void 0})}
function kv(a){a=ev(a===void 0?0:a);if(!a&&!R("USE_CSN_FALLBACK",!0))return null;a||(a="UNDEFINED_CSN");return a?a:null}
E("yt_logging_screen.getCurrentCsn",kv);function lv(a,b,c){var d=iv();(c=kv(c))&&delete d[c];b&&(d[a]=b)}
function mv(a){return iv()[a]}
E("yt_logging_screen.getCttAuthInfo",mv);E("yt_logging_screen.setCurrentScreen",function(a,b,c,d){c=c===void 0?0:c;if(a!==ev(c)||b!==R(gv(c)))if(lv(a,d,c),fv(a,c),Lm(gv(c),b),b=function(){setTimeout(function(){a&&dp("foregroundHeartbeatScreenAssociated",{clientDocumentNonce:bv,clientScreenNonce:a})},0)},"requestAnimationFrame"in window)try{window.requestAnimationFrame(b)}catch(e){b()}else b()});function nv(){var a=Cg(ov),b;return(new wi(function(c,d){a.onSuccess=function(e){gn(e)?c(new pv(e)):d(new qv("Request failed, status="+(e&&"status"in e?e.status:-1),"net.badstatus",e))};
a.onError=function(e){d(new qv("Unknown request error","net.unknown",e))};
a.onTimeout=function(e){d(new qv("Request timed out","net.timeout",e))};
b=un("//googleads.g.doubleclick.net/pagead/id",a)})).Gc(function(c){if(c instanceof Fi){var d;
(d=b)==null||d.abort()}return Bi(c)})}
function qv(a,b,c){fb.call(this,a+", errorCode="+b);this.errorCode=b;this.xhr=c;this.name="PromiseAjaxError"}
v(qv,fb);function pv(a){this.xhr=a}
;function rv(){this.X=0;this.h=null}
rv.prototype.then=function(a,b,c){return this.X===1&&a?(a=a.call(c,this.h))&&typeof a.then==="function"?a:sv(a):this.X===2&&b?(a=b.call(c,this.h))&&typeof a.then==="function"?a:tv(a):this};
rv.prototype.getValue=function(){return this.h};
rv.prototype.isRejected=function(){return this.X==2};
rv.prototype.$goog_Thenable=!0;function tv(a){var b=new rv;a=a===void 0?null:a;b.X=2;b.h=a===void 0?null:a;return b}
function sv(a){var b=new rv;a=a===void 0?null:a;b.X=1;b.h=a===void 0?null:a;return b}
;function uv(a){var b=R("INNERTUBE_HOST_OVERRIDE");b&&(a=String(b)+String(mc(a)));return a}
function vv(a){var b={};T("json_condensed_response")&&(b.prettyPrint="false");return a=an(a,b||{},!1)}
function wv(a,b){var c=c===void 0?{}:c;a={method:b===void 0?"POST":b,mode:bn(a)?"same-origin":"cors",credentials:bn(a)?"same-origin":"include"};b={};for(var d=y(Object.keys(c)),e=d.next();!e.done;e=d.next())e=e.value,c[e]&&(b[e]=c[e]);Object.keys(b).length>0&&(a.headers=b);return a}
;function xv(){return kg()||(qd||rd)&&jp("applewebkit")&&!jp("version")&&(!jp("safari")||jp("gsa/"))||pd&&jp("version/")?!0:R("EOM_VISITOR_DATA")?!1:!0}
;function yv(a){var b=a.docid||a.video_id||a.videoId||a.id;if(b)return b;b=a.raw_player_response;b||(a=a.player_response)&&(b=JSON.parse(a));return b&&b.videoDetails&&b.videoDetails.videoId||null}
;function zv(a){a:{var b="EMBEDDED_PLAYER_MODE_UNKNOWN";window.location.hostname.includes("youtubeeducation.com")&&(b="EMBEDDED_PLAYER_MODE_PFL");var c=a.raw_embedded_player_response;if(!c&&(a=a.embedded_player_response))try{c=JSON.parse(a)}catch(e){break a}if(c)b:for(var d in rm)if(rm[d]==c.embeddedPlayerMode){b=rm[d];break b}}return b==="EMBEDDED_PLAYER_MODE_PFL"}
;function Av(a){fb.call(this,a.message||a.description||a.name);this.isMissing=a instanceof Bv;this.isTimeout=a instanceof qv&&a.errorCode=="net.timeout";this.isCanceled=a instanceof Fi}
v(Av,fb);Av.prototype.name="BiscottiError";function Bv(){fb.call(this,"Biscotti ID is missing from server")}
v(Bv,fb);Bv.prototype.name="BiscottiMissingError";var ov={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},Cv=null;function Dv(){if(T("disable_biscotti_fetch_entirely_for_all_web_clients"))return Error("Biscotti id fetching has been disabled entirely.");if(!xv())return Error("User has not consented - not fetching biscotti id.");var a=R("PLAYER_VARS",{});if(Ag(a)=="1")return Error("Biscotti ID is not available in private embed mode");if(zv(a))return Error("Biscotti id fetching has been disabled for pfl.")}
function Em(){var a=Dv();if(a!==void 0)return Bi(a);Cv||(Cv=nv().then(Ev).Gc(function(b){return Fv(2,b)}));
return Cv}
function Ev(a){a=a.xhr.responseText;if(a.lastIndexOf(")]}'",0)!=0)throw new Bv;a=JSON.parse(a.substr(4));if((a.type||1)>1)throw new Bv;a=a.id;Fm(a);Cv=sv(a);Gv(18E5,2);return a}
function Fv(a,b){b=new Av(b);Fm("");Cv=tv(b);a>0&&Gv(12E4,a-1);throw b;}
function Gv(a,b){hn(function(){nv().then(Ev,function(c){return Fv(b,c)}).Gc(ui)},a)}
function Hv(){try{var a=F("yt.ads.biscotti.getId_");return a?a():Em()}catch(b){return Bi(b)}}
;var Nb=oa(["data-"]);function Iv(a){a&&(a.dataset?a.dataset[Jv()]="true":Ob(a))}
function Kv(a){return a?a.dataset?a.dataset[Jv()]:a.getAttribute("data-loaded"):null}
var Lv={};function Jv(){return Lv.loaded||(Lv.loaded="loaded".replace(/\-([a-z])/g,function(a,b){return b.toUpperCase()}))}
;function Mv(a){a=a||{};var b={},c={};this.url=a.url||"";this.args=a.args||Cg(b);this.assets=a.assets||{};this.attrs=a.attrs||Cg(c);this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
Mv.prototype.clone=function(){var a=new Mv,b;for(b in this)if(this.hasOwnProperty(b)){var c=this[b];La(c)=="object"?a[b]=Cg(c):a[b]=c}return a};var Nv=["att/get"],Ov=["share/get_share_panel"],Pv=["share/get_web_player_share_panel"],Qv=["feedback"],Rv=["notification/modify_channel_preference"],Sv=["browse/edit_playlist"],Tv=["subscription/subscribe"],Uv=["subscription/unsubscribe"];var Vv=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};E("yt.msgs_",Vv);function Wv(a){Gm(Vv,arguments)}
;function Xv(a,b,c){Yv(a,b,c===void 0?null:c)}
function Zv(a){a=$v(a);var b=document.getElementById(a);b&&(rt(a),b.parentNode.removeChild(b))}
function aw(a,b){a&&b&&(a=""+Pa(b),(a=bw[a])&&pt(a))}
function Yv(a,b,c){c=c===void 0?null:c;var d=$v(a),e=document.getElementById(d),f=e&&Kv(e),g=e&&!f;f?b&&b():(b&&(f=nt(d,b),b=""+Pa(b),bw[b]=f),g||(e=cw(a,d,function(){Kv(e)||(Iv(e),qt(d),hn(function(){rt(d)},0))},c)))}
function cw(a,b,c,d){d=d===void 0?null:d;var e=Jg("SCRIPT");e.id=b;e.onload=function(){c&&setTimeout(c,0)};
e.onreadystatechange=function(){switch(e.readyState){case "loaded":case "complete":e.onload()}};
d&&e.setAttribute("nonce",d);Lb(e,im(a));a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(e,a.firstChild);return e}
function $v(a){var b=document.createElement("a");Db(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+hc(a)}
var bw={};function dw(a){var b=ew(a),c=document.getElementById(b),d=c&&Kv(c);d||c&&!d||(c=fw(a,b,function(){if(!Kv(c)){Iv(c);qt(b);var e=Va(rt,b);hn(e,0)}}))}
function fw(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=im(a);Qb(d,a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function ew(a){var b=Jg("A");Db(b,new vb(a));a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+hc(a)}
;function gw(a){var b=C.apply(1,arguments);if(!hw(a)||b.some(function(d){return!hw(d)}))throw Error("Only objects may be merged.");
b=y(b);for(var c=b.next();!c.done;c=b.next())iw(a,c.value)}
function iw(a,b){for(var c in b)if(hw(b[c])){if(c in a&&!hw(a[c]))throw Error("Cannot merge an object into a non-object.");c in a||(a[c]={});iw(a[c],b[c])}else if(jw(b[c])){if(c in a&&!jw(a[c]))throw Error("Cannot merge an array into a non-array.");c in a||(a[c]=[]);kw(a[c],b[c])}else a[c]=b[c];return a}
function kw(a,b){b=y(b);for(var c=b.next();!c.done;c=b.next())c=c.value,hw(c)?a.push(iw({},c)):jw(c)?a.push(kw([],c)):a.push(c);return a}
function hw(a){return typeof a==="object"&&!Array.isArray(a)}
function jw(a){return typeof a==="object"&&Array.isArray(a)}
;var lw="absolute_experiments app conditional_experiments debugcss debugjs expflag forced_experiments pbj pbjreload sbb spf spfreload sr_bns_address sttick".split(" ");
function mw(a,b){var c=c===void 0?!0:c;var d=R("VALID_SESSION_TEMPDATA_DOMAINS",[]),e=kc(window.location.href);e&&d.push(e);e=kc(a);if(Sb(d,e)>=0||!e&&a.lastIndexOf("/",0)==0)if(d=document.createElement("a"),Db(d,a),a=d.href)if(a=mc(a),a=nc(a))if(c&&!b.csn&&(b.itct||b.ved)&&(b=Object.assign({csn:kv()},b)),f){var f=parseInt(f,10);isFinite(f)&&f>0&&nw(a,b,f)}else nw(a,b)}
function nw(a,b,c){a=ow(a);b=b?qc(b):"";c=c||5;xv()&&Vn(a,b,c)}
function ow(a){for(var b=y(lw),c=b.next();!c.done;c=b.next())a=vc(a,c.value);return"ST-"+hc(a).toString(36)}
;function pw(a){$q.call(this,1,arguments);this.csn=a}
v(pw,$q);var ir=new ar("screen-created",pw),qw=[],rw=0,sw=new Map,tw=new Map,uw=new Map;
function vw(a,b,c,d,e){e=e===void 0?!1:e;for(var f=ww({cttAuthInfo:mv(b)||void 0},b),g=y(d),h=g.next();!h.done;h=g.next()){h=h.value;var k=h.getAsJson();(yg(k)||!k.trackingParams&&!k.veType)&&Tu(Error("Child VE logged with no data"));if(T("no_client_ve_attach_unless_shown")){var l=xw(h,b);if(k.veType&&!tw.has(l)&&!uw.has(l)&&!e){if(!T("il_attach_cache_limit")||sw.size<1E3){sw.set(l,[a,b,c,h]);return}T("il_attach_cache_limit")&&sw.size>1E3&&Tu(new U("IL Attach cache exceeded limit"))}h=xw(c,b);sw.has(h)?
yw(c,b):uw.set(h,!0)}}d=d.filter(function(m){m.csn!==b?(m.csn=b,m=!0):m=!1;return m});
c={csn:b,parentVe:c.getAsJson(),childVes:Vb(d,function(m){return m.getAsJson()})};
b==="UNDEFINED_CSN"?zw("visualElementAttached",f,c):a?Ku("visualElementAttached",c,a,f):dp("visualElementAttached",c,f)}
function zw(a,b,c){qw.push({Te:a,payload:c,Qh:void 0,options:b});rw||(rw=jr())}
function kr(a){if(qw){for(var b=y(qw),c=b.next();!c.done;c=b.next())c=c.value,c.payload&&(c.payload.csn=a.csn,dp(c.Te,c.payload,c.options));qw.length=0}rw=0}
function xw(a,b){return""+a.getAsJson().veType+a.getAsJson().veCounter+b}
function yw(a,b){a=xw(a,b);sw.has(a)&&(b=sw.get(a)||[],vw(b[0],b[1],b[2],[b[3]],!0),sw.delete(a))}
function ww(a,b){T("log_sequence_info_on_gel_web")&&(a.sequenceGroup=b);return a}
;function Aw(){try{return!!self.localStorage}catch(a){return!1}}
;function Bw(a){a=a.match(/(.*)::.*::.*/);if(a!==null)return a[1]}
function Cw(a){if(Aw()){var b=Object.keys(window.localStorage);b=y(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=Bw(c);d===void 0||a.includes(d)||self.localStorage.removeItem(c)}}}
function Dw(){if(!Aw())return!1;var a=no(),b=Object.keys(window.localStorage);b=y(b);for(var c=b.next();!c.done;c=b.next())if(c=Bw(c.value),c!==void 0&&c!==a)return!0;return!1}
;function Ew(){var a=!1;try{a=!!window.sessionStorage.getItem("session_logininfo")}catch(b){a=!0}return(R("INNERTUBE_CLIENT_NAME")==="WEB"||R("INNERTUBE_CLIENT_NAME")==="WEB_CREATOR")&&a}
function Fw(){var a=a===void 0?!0:a;try{window.sessionStorage.removeItem("stickiness_reload");window.sessionStorage.removeItem("session_logininfo");Lm("LOGIN_INFO","");a&&window.sessionStorage.setItem("from_switch_account","1");a=!0;a=a===void 0?!1:a;var b,c=Gw;c||(c=document.querySelector("#persist_identity"));if(b=c){var d=b.src?(new URL(b.src)).origin:"*";if(a){var e;(e=b.contentWindow)==null||e.postMessage({action:"clear"},d)}else if(!(Number(window.sessionStorage.getItem("stickiness_reload"))>=
2)){var f=window.sessionStorage.getItem("session_logininfo");if(f){var g;(g=b.contentWindow)==null||g.postMessage({loginInfo:f},d)}}}}catch(h){}}
function Hw(a){if(a)if(a.startsWith("https://accounts.google.com/AddSession"))Fw();else if(a.startsWith("https://accounts.google.com/ServiceLogin"))Fw();else{var b;if(b=a.startsWith("https://myaccount.google.com"))b=(a instanceof Dk?a.clone():new Dk(a)).h.endsWith("/youtubeoptions");b&&Fw()}if(R("LOGGED_IN",!0)&&Ew()){b=R("VALID_SESSION_TEMPDATA_DOMAINS",[]);var c=kc(window.location.href);c&&b.push(c);c=kc(a);Sb(b,c)>=0||!c&&a.lastIndexOf("/",0)==0?(b=mc(a),(b=nc(b))?(b=ow(b),b=(b=Wn(b)||null)?Ym(b):
{}):b=null):b=null;b==null&&(b={});c=b;var d=void 0;Ew()?(d||(d=R("LOGIN_INFO")),d?(c.session_logininfo=d,c=!0):c=!1):c=!1;c&&mw(a,b)}}
var Gw=null;function Iw(a,b,c){b=b===void 0?{}:b;c=c===void 0?!1:c;var d=R("EVENT_ID");d&&(b.ei||(b.ei=d));b&&mw(a,b);if(c)return!1;Hw(a);var e=e===void 0?{}:e;var f=f===void 0?"":f;var g=g===void 0?window:g;b=rc(a,e);Hw(b);a=void 0;a=a===void 0?zb:a;a:if(f=b+f,a=a===void 0?zb:a,!(f instanceof vb)){for(b=0;b<a.length;++b)if(c=a[b],c instanceof xb&&c.Ge(f)){f=new vb(f);break a}f=void 0}g=g.location;f=Bb(f||wb);f!==void 0&&(g.href=f);return!0}
;function Jw(a){if(Ag(R("PLAYER_VARS",{}))!="1"){a&&Dm();try{Hv().then(function(){},function(){}),hn(Jw,18E5)}catch(b){Qm(b)}}}
;var Kw=new Map([["dark","USER_INTERFACE_THEME_DARK"],["light","USER_INTERFACE_THEME_LIGHT"]]);function Lw(){var a=a===void 0?window.location.href:a;if(T("kevlar_disable_theme_param"))return null;var b=lc(a);if(T("enable_dark_theme_only_on_shorts")&&b!=null&&b.startsWith("/shorts/"))return"USER_INTERFACE_THEME_DARK";try{var c=Zm(a).theme;return Kw.get(c)||null}catch(d){}return null}
;function Mw(){this.h={};if(this.i=Yn()){var a=Wn("CONSISTENCY");a&&Nw(this,{encryptedTokenJarContents:a})}}
Mw.prototype.handleResponse=function(a,b){if(!b)throw Error("request needs to be passed into ConsistencyService");var c,d;b=((c=b.Ga.context)==null?void 0:(d=c.request)==null?void 0:d.consistencyTokenJars)||[];var e;if(a=(e=a.responseContext)==null?void 0:e.consistencyTokenJar){e=y(b);for(c=e.next();!c.done;c=e.next())delete this.h[c.value.encryptedTokenJarContents];Nw(this,a)}};
function Nw(a,b){if(b.encryptedTokenJarContents&&(a.h[b.encryptedTokenJarContents]=b,typeof b.expirationSeconds==="string")){var c=Number(b.expirationSeconds);setTimeout(function(){delete a.h[b.encryptedTokenJarContents]},c*1E3);
a.i&&Vn("CONSISTENCY",b.encryptedTokenJarContents,c,void 0,!0)}}
;var Ow=window.location.hostname.split(".").slice(-2).join(".");function Pw(){this.j=-1;var a=R("LOCATION_PLAYABILITY_TOKEN");R("INNERTUBE_CLIENT_NAME")==="TVHTML5"&&(this.h=Qw(this))&&(a=this.h.get("yt-location-playability-token"));a&&(this.locationPlayabilityToken=a,this.i=void 0)}
var Rw;function Sw(){Rw=F("yt.clientLocationService.instance");Rw||(Rw=new Pw,E("yt.clientLocationService.instance",Rw));return Rw}
p=Pw.prototype;
p.setLocationOnInnerTubeContext=function(a){a.client||(a.client={});if(this.i)a.client.locationInfo||(a.client.locationInfo={}),a.client.locationInfo.latitudeE7=Math.floor(this.i.coords.latitude*1E7),a.client.locationInfo.longitudeE7=Math.floor(this.i.coords.longitude*1E7),a.client.locationInfo.horizontalAccuracyMeters=Math.round(this.i.coords.accuracy),a.client.locationInfo.forceLocationPlayabilityTokenRefresh=!0;else if(this.o||this.locationPlayabilityToken)a.client.locationPlayabilityToken=this.o||
this.locationPlayabilityToken};
p.handleResponse=function(a){var b;a=(b=a.responseContext)==null?void 0:b.locationPlayabilityToken;a!==void 0&&(this.locationPlayabilityToken=a,this.i=void 0,R("INNERTUBE_CLIENT_NAME")==="TVHTML5"?(this.h=Qw(this))&&this.h.set("yt-location-playability-token",a,15552E3):Vn("YT_CL",JSON.stringify({loctok:a}),15552E3,Ow,!0))};
function Qw(a){return a.h===void 0?new Vo("yt-client-location"):a.h}
p.clearLocationPlayabilityToken=function(a){a==="TVHTML5"?(this.h=Qw(this))&&this.h.remove("yt-location-playability-token"):Xn("YT_CL");this.o=void 0;this.j!==-1&&(clearTimeout(this.j),this.j=-1)};
p.getCurrentPositionFromGeolocation=function(){var a=this;if(!(navigator&&navigator.geolocation&&navigator.geolocation.getCurrentPosition))return Promise.reject(Error("Geolocation unsupported"));var b=!1,c=1E4;R("INNERTUBE_CLIENT_NAME")==="MWEB"&&(b=!0,c=15E3);return new Promise(function(d,e){navigator.geolocation.getCurrentPosition(function(f){a.i=f;d(f)},function(f){e(f)},{enableHighAccuracy:b,
maximumAge:0,timeout:c})})};
p.createUnpluggedLocationInfo=function(a){var b={};a=a.coords;if(a==null?0:a.latitude)b.latitudeE7=Math.floor(a.latitude*1E7);if(a==null?0:a.longitude)b.longitudeE7=Math.floor(a.longitude*1E7);if(a==null?0:a.accuracy)b.locationRadiusMeters=Math.round(a.accuracy);return b};
p.createLocationInfo=function(a){var b={};a=a.coords;if(a==null?0:a.latitude)b.latitudeE7=Math.floor(a.latitude*1E7);if(a==null?0:a.longitude)b.longitudeE7=Math.floor(a.longitude*1E7);return b};function Tw(a,b,c){b=b===void 0?!1:b;c=c===void 0?!1:c;var d=R("INNERTUBE_CONTEXT");if(!d)return Su(Error("Error: No InnerTubeContext shell provided in ytconfig.")),{};d=Dg(d);T("web_no_tracking_params_in_shell_killswitch")||delete d.clickTracking;d.client||(d.client={});var e=d.client;e.clientName==="MWEB"&&e.clientFormFactor!=="AUTOMOTIVE_FORM_FACTOR"&&(e.clientFormFactor=R("IS_TABLET")?"LARGE_FORM_FACTOR":"SMALL_FORM_FACTOR");e.screenWidthPoints=window.innerWidth;e.screenHeightPoints=window.innerHeight;
e.screenPixelDensity=Math.round(window.devicePixelRatio||1);e.screenDensityFloat=window.devicePixelRatio||1;e.utcOffsetMinutes=-Math.floor((new Date).getTimezoneOffset());var f=f===void 0?!1:f;bo();var g="USER_INTERFACE_THEME_LIGHT";fo(165)?g="USER_INTERFACE_THEME_DARK":fo(174)?g="USER_INTERFACE_THEME_LIGHT":!T("kevlar_legacy_browsers")&&window.matchMedia&&window.matchMedia("(prefers-color-scheme)").matches&&window.matchMedia("(prefers-color-scheme: dark)").matches&&(g="USER_INTERFACE_THEME_DARK");
f=f?g:Lw()||g;e.userInterfaceTheme=f;if(!b){if(f=ko())e.connectionType=f;T("web_log_effective_connection_type")&&(f=lo())&&(d.client.effectiveConnectionType=f)}var h;if(T("web_log_memory_total_kbytes")&&((h=D.navigator)==null?0:h.deviceMemory)){var k;h=(k=D.navigator)==null?void 0:k.deviceMemory;d.client.memoryTotalKbytes=""+h*1E6}T("web_gcf_hashes_innertube")&&(f=Rq())&&(k=f.coldConfigData,h=f.coldHashData,f=f.hotHashData,d.client.configInfo=d.client.configInfo||{},k&&(d.client.configInfo.coldConfigData=
k),h&&(d.client.configInfo.coldHashData=h),f&&(d.client.configInfo.hotHashData=f));k=Zm(D.location.href);!T("web_populate_internal_geo_killswitch")&&k.internalcountrycode&&(e.internalGeo=k.internalcountrycode);e.clientName==="MWEB"||e.clientName==="WEB"?(e.mainAppWebInfo={graftUrl:D.location.href},T("kevlar_woffle")&&Pn.instance&&(k=Pn.instance,e.mainAppWebInfo.pwaInstallabilityStatus=!k.h&&k.i?"PWA_INSTALLABILITY_STATUS_CAN_BE_INSTALLED":"PWA_INSTALLABILITY_STATUS_UNKNOWN"),e.mainAppWebInfo.webDisplayMode=
Qn(),e.mainAppWebInfo.isWebNativeShareAvailable=navigator&&navigator.share!==void 0):e.clientName==="TVHTML5"&&(!T("web_lr_app_quality_killswitch")&&(k=R("LIVING_ROOM_APP_QUALITY"))&&(e.tvAppInfo=Object.assign(e.tvAppInfo||{},{appQuality:k})),k=R("LIVING_ROOM_CERTIFICATION_SCOPE"))&&(e.tvAppInfo=Object.assign(e.tvAppInfo||{},{certificationScope:k}));if(!T("web_populate_time_zone_itc_killswitch")){a:{if(typeof Intl!=="undefined")try{var l=(new Intl.DateTimeFormat).resolvedOptions().timeZone;break a}catch(Z){}l=
void 0}l&&(e.timeZone=l)}(l=R("EXPERIMENTS_TOKEN",""))?e.experimentsToken=l:delete e.experimentsToken;l=mn();Mw.instance||(Mw.instance=new Mw);d.request=Object.assign({},d.request,{internalExperimentFlags:l,consistencyTokenJars:vg(Mw.instance.h)});!T("web_prequest_context_killswitch")&&(l=R("INNERTUBE_CONTEXT_PREQUEST_CONTEXT"))&&(d.request.externalPrequestContext=l);e=bo();l=fo(58);e=e.get("gsml","");d.user=Object.assign({},d.user);l&&(d.user.enableSafetyMode=l);e&&(d.user.lockedSafetyMode=!0);T("warm_op_csn_cleanup")?
c&&(b=kv())&&(d.clientScreenNonce=b):!b&&(b=kv())&&(d.clientScreenNonce=b);a&&(d.clickTracking={clickTrackingParams:a});if(a=F("yt.mdx.remote.remoteClient_"))d.remoteClient=a;Sw().setLocationOnInnerTubeContext(d);try{var m=cn(),n=m.bid;delete m.bid;d.adSignalsInfo={params:[],bid:n};for(var r=y(Object.entries(m)),t=r.next();!t.done;t=r.next()){var w=y(t.value),x=w.next().value,z=w.next().value;m=x;n=z;a=void 0;(a=d.adSignalsInfo.params)==null||a.push({key:m,value:""+n})}var G,H;if(((G=d.client)==null?
void 0:G.clientName)==="TVHTML5"||((H=d.client)==null?void 0:H.clientName)==="TVHTML5_UNPLUGGED"){var S=R("INNERTUBE_CONTEXT");S.adSignalsInfo&&(d.adSignalsInfo.advertisingId=S.adSignalsInfo.advertisingId,d.adSignalsInfo.advertisingIdSignalType="DEVICE_ID_TYPE_CONNECTED_TV_IFA",d.adSignalsInfo.limitAdTracking=S.adSignalsInfo.limitAdTracking)}}catch(Z){Su(Z)}return d}
;function Uw(a){var b={"Content-Type":"application/json"};R("EOM_VISITOR_DATA")?b["X-Goog-EOM-Visitor-Id"]=R("EOM_VISITOR_DATA"):R("VISITOR_DATA")&&(b["X-Goog-Visitor-Id"]=R("VISITOR_DATA"));b["X-Youtube-Bootstrap-Logged-In"]=R("LOGGED_IN",!1);R("DEBUG_SETTINGS_METADATA")&&(b["X-Debug-Settings-Metadata"]=R("DEBUG_SETTINGS_METADATA"));a!=="cors"&&((a=R("INNERTUBE_CONTEXT_CLIENT_NAME"))&&(b["X-Youtube-Client-Name"]=a),(a=R("INNERTUBE_CONTEXT_CLIENT_VERSION"))&&(b["X-Youtube-Client-Version"]=a),(a=R("CHROME_CONNECTED_HEADER"))&&
(b["X-Youtube-Chrome-Connected"]=a),(a=R("DOMAIN_ADMIN_STATE"))&&(b["X-Youtube-Domain-Admin-State"]=a),R("ENABLE_LAVA_HEADER_ON_IT_EXPANSION")&&(a=R("SERIALIZED_LAVA_DEVICE_CONTEXT"))&&(b["X-YouTube-Lava-Device-Context"]=a));return b}
;function Vw(){this.h={}}
p=Vw.prototype;p.contains=function(a){return Object.prototype.hasOwnProperty.call(this.h,a)};
p.get=function(a){if(this.contains(a))return this.h[a]};
p.set=function(a,b){this.h[a]=b};
p.Tb=function(){return Object.keys(this.h)};
p.remove=function(a){delete this.h[a]};function Ww(){this.mappings=new Vw}
Ww.prototype.getModuleId=function(a){return a.serviceId.getModuleId()};
Ww.prototype.get=function(a){var b=this.mappings.get(a.toString());a:switch(b.type){case "mapping":a=b.value;break a;case "factory":b=b.value();this.mappings.set(a.toString(),{type:"mapping",value:b});a=b;break a;default:a=Eb(b)}return a};
new Ww;function Xw(a){return function(){return new a}}
;var Yw={},Zw=(Yw.WEB_UNPLUGGED="^unplugged/",Yw.WEB_UNPLUGGED_ONBOARDING="^unplugged/",Yw.WEB_UNPLUGGED_OPS="^unplugged/",Yw.WEB_UNPLUGGED_PUBLIC="^unplugged/",Yw.WEB_CREATOR="^creator/",Yw.WEB_KIDS="^kids/",Yw.WEB_EXPERIMENTS="^experiments/",Yw.WEB_MUSIC="^music/",Yw.WEB_REMIX="^music/",Yw.WEB_MUSIC_EMBEDDED_PLAYER="^music/",Yw.WEB_MUSIC_EMBEDDED_PLAYER="^main_app/|^sfv/",Yw);
function $w(a){var b=b===void 0?"UNKNOWN_INTERFACE":b;if(a.length===1)return a[0];var c=Zw[b];if(c){c=new RegExp(c);for(var d=y(a),e=d.next();!e.done;e=d.next())if(e=e.value,c.exec(e))return e}var f=[];Object.entries(Zw).forEach(function(g){var h=y(g);g=h.next().value;h=h.next().value;b!==g&&f.push(h)});
c=new RegExp(f.join("|"));a.sort(function(g,h){return g.length-h.length});
d=y(a);for(e=d.next();!e.done;e=d.next())if(e=e.value,!c.exec(e))return e;return a[0]}
;function ax(){}
ax.prototype.u=function(a,b,c){b=b===void 0?{}:b;c=c===void 0?Un:c;var d={context:Tw(a.clickTrackingParams,!1,this.o)};var e=this.i(a);if(e){this.h(d,e,b);var f;b="/youtubei/v1/"+$w(this.j());(e=(f=bu(a.commandMetadata,pm))==null?void 0:f.apiUrl)&&(b=e);f=vv(uv(b));a=Object.assign({},{command:a},void 0);d={input:f,ab:wv(f),Ga:d,config:a};d.config.Pb?d.config.Pb.identity=c:d.config.Pb={identity:c};return d}c=new U("Error: Failed to create Request from Command.",a);Su(c)};
da.Object.defineProperties(ax.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!1}}});
function bx(){}
v(bx,ax);function cx(){}
v(cx,bx);cx.prototype.u=function(){return{input:"/getDatasyncIdsEndpoint",ab:wv("/getDatasyncIdsEndpoint","GET"),Ga:{}}};
cx.prototype.j=function(){return[]};
cx.prototype.i=function(){};
cx.prototype.h=function(){};var dx={},ex=(dx.GET_DATASYNC_IDS=Xw(cx),dx);function fx(a){var b;(b=F("ytcsi."+(a||"")+"data_"))||(b={tick:{},info:{}},E("ytcsi."+(a||"")+"data_",b));return b}
function gx(){var a=fx();a.info||(a.info={});return a.info}
function hx(a){a=fx(a);a.metadata||(a.metadata={});return a.metadata}
function ix(a){a=fx(a);a.tick||(a.tick={});return a.tick}
function jx(a){a=fx(a);if(a.gel){var b=a.gel;b.gelInfos||(b.gelInfos={});b.gelTicks||(b.gelTicks={})}else a.gel={gelTicks:{},gelInfos:{}};return a.gel}
function kx(a){a=jx(a);a.gelInfos||(a.gelInfos={});return a.gelInfos}
function lx(a){var b=fx(a).nonce;b||(b=av(),fx(a).nonce=b);return b}
;function mx(){var a=F("ytcsi.debug");a||(a=[],E("ytcsi.debug",a),E("ytcsi.reference",{}));return a}
function nx(a){a=a||"";var b=F("ytcsi.reference");b||(mx(),b=F("ytcsi.reference"));if(b[a])return b[a];var c=mx(),d={timerName:a,info:{},tick:{},span:{},jspbInfo:[]};c.push(d);return b[a]=d}
;var W={},ox=(W.auto_search="LATENCY_ACTION_AUTO_SEARCH",W.ad_to_ad="LATENCY_ACTION_AD_TO_AD",W.ad_to_video="LATENCY_ACTION_AD_TO_VIDEO",W.app_startup="LATENCY_ACTION_APP_STARTUP",W.browse="LATENCY_ACTION_BROWSE",W.cast_splash="LATENCY_ACTION_CAST_SPLASH",W.channel_activity="LATENCY_ACTION_KIDS_CHANNEL_ACTIVITY",W.channels="LATENCY_ACTION_CHANNELS",W.chips="LATENCY_ACTION_CHIPS",W.commerce_transaction="LATENCY_ACTION_COMMERCE_TRANSACTION",W.direct_playback="LATENCY_ACTION_DIRECT_PLAYBACK",W.editor=
"LATENCY_ACTION_EDITOR",W.embed="LATENCY_ACTION_EMBED",W.embed_no_video="LATENCY_ACTION_EMBED_NO_VIDEO",W.entity_key_serialization_perf="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF",W.entity_key_deserialization_perf="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF",W.explore="LATENCY_ACTION_EXPLORE",W.favorites="LATENCY_ACTION_FAVORITES",W.home="LATENCY_ACTION_HOME",W.inboarding="LATENCY_ACTION_INBOARDING",W.landing="LATENCY_ACTION_LANDING",W.library="LATENCY_ACTION_LIBRARY",W.live="LATENCY_ACTION_LIVE",
W.live_pagination="LATENCY_ACTION_LIVE_PAGINATION",W.management="LATENCY_ACTION_MANAGEMENT",W.mini_app="LATENCY_ACTION_MINI_APP_PLAY",W.notification_settings="LATENCY_ACTION_KIDS_NOTIFICATION_SETTINGS",W.onboarding="LATENCY_ACTION_ONBOARDING",W.parent_profile_settings="LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS",W.parent_tools_collection="LATENCY_ACTION_PARENT_TOOLS_COLLECTION",W.parent_tools_dashboard="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD",W.player_att="LATENCY_ACTION_PLAYER_ATTESTATION",W.prebuffer=
"LATENCY_ACTION_PREBUFFER",W.prefetch="LATENCY_ACTION_PREFETCH",W.profile_settings="LATENCY_ACTION_KIDS_PROFILE_SETTINGS",W.profile_switcher="LATENCY_ACTION_LOGIN",W.projects="LATENCY_ACTION_PROJECTS",W.reel_watch="LATENCY_ACTION_REEL_WATCH",W.results="LATENCY_ACTION_RESULTS",W.red="LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE",W.premium="LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE",W.privacy_policy="LATENCY_ACTION_KIDS_PRIVACY_POLICY",W.review="LATENCY_ACTION_REVIEW",W.search_overview_answer="LATENCY_ACTION_SEARCH_OVERVIEW_ANSWER",
W.search_ui="LATENCY_ACTION_SEARCH_UI",W.search_suggest="LATENCY_ACTION_SUGGEST",W.search_zero_state="LATENCY_ACTION_SEARCH_ZERO_STATE",W.secret_code="LATENCY_ACTION_KIDS_SECRET_CODE",W.seek="LATENCY_ACTION_PLAYER_SEEK",W.settings="LATENCY_ACTION_SETTINGS",W.store="LATENCY_ACTION_STORE",W.supervision_dashboard="LATENCY_ACTION_KIDS_SUPERVISION_DASHBOARD",W.tenx="LATENCY_ACTION_TENX",W.video_preview="LATENCY_ACTION_VIDEO_PREVIEW",W.video_to_ad="LATENCY_ACTION_VIDEO_TO_AD",W.watch="LATENCY_ACTION_WATCH",
W.watch_it_again="LATENCY_ACTION_KIDS_WATCH_IT_AGAIN",W["watch,watch7"]="LATENCY_ACTION_WATCH",W["watch,watch7_html5"]="LATENCY_ACTION_WATCH",W["watch,watch7ad"]="LATENCY_ACTION_WATCH",W["watch,watch7ad_html5"]="LATENCY_ACTION_WATCH",W.wn_comments="LATENCY_ACTION_LOAD_COMMENTS",W.ww_rqs="LATENCY_ACTION_WHO_IS_WATCHING",W.voice_assistant="LATENCY_ACTION_VOICE_ASSISTANT",W.cast_load_by_entity_to_watch="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH",W.networkless_performance="LATENCY_ACTION_NETWORKLESS_PERFORMANCE",
W.gel_compression="LATENCY_ACTION_GEL_COMPRESSION",W.gel_jspb_serialize="LATENCY_ACTION_GEL_JSPB_SERIALIZE",W.attestation_challenge_fetch="LATENCY_ACTION_ATTESTATION_CHALLENGE_FETCH",W);function px(a,b){$q.call(this,1,arguments);this.timer=b}
v(px,$q);var qx=new ar("aft-recorded",px);E("ytLoggingGelSequenceIdObj_",D.ytLoggingGelSequenceIdObj_||{});var rx=D.ytLoggingLatencyUsageStats_||{};E("ytLoggingLatencyUsageStats_",rx);function sx(){this.h=0}
function tx(){sx.instance||(sx.instance=new sx);return sx.instance}
sx.prototype.tick=function(a,b,c,d){ux(this,"tick_"+a+"_"+b)||dp("latencyActionTicked",{tickName:a,clientActionNonce:b},{timestamp:c,cttAuthInfo:d})};
sx.prototype.info=function(a,b,c){var d=Object.keys(a).join("");ux(this,"info_"+d+"_"+b)||(a=Object.assign({},a),a.clientActionNonce=b,dp("latencyActionInfo",a,{cttAuthInfo:c}))};
sx.prototype.jspbInfo=function(){};
sx.prototype.span=function(a,b,c){var d=Object.keys(a).join("");ux(this,"span_"+d+"_"+b)||(a.clientActionNonce=b,dp("latencyActionSpan",a,{cttAuthInfo:c}))};
function ux(a,b){rx[b]=rx[b]||{count:0};var c=rx[b];c.count++;c.time=V();a.h||(a.h=po(function(){var d=V(),e;for(e in rx)rx[e]&&d-rx[e].time>6E4&&delete rx[e];a&&(a.h=0)},5E3));
return c.count>5?(c.count===6&&Math.random()*1E5<1&&(c=new U("CSI data exceeded logging limit with key",b.split("_")),b.indexOf("plev")>=0||Tu(c)),!0):!1}
;var vx=window;function wx(){this.timing={};this.clearResourceTimings=function(){};
this.webkitClearResourceTimings=function(){};
this.mozClearResourceTimings=function(){};
this.msClearResourceTimings=function(){};
this.oClearResourceTimings=function(){}}
function xx(){var a;if(T("csi_use_performance_navigation_timing")||T("csi_use_performance_navigation_timing_tvhtml5")){var b,c,d,e=X==null?void 0:(a=X.getEntriesByType)==null?void 0:(b=a.call(X,"navigation"))==null?void 0:(c=b[0])==null?void 0:(d=c.toJSON)==null?void 0:d.call(c);e?(e.requestStart=yx(e.requestStart),e.responseEnd=yx(e.responseEnd),e.redirectStart=yx(e.redirectStart),e.redirectEnd=yx(e.redirectEnd),e.domainLookupEnd=yx(e.domainLookupEnd),e.connectStart=yx(e.connectStart),e.connectEnd=
yx(e.connectEnd),e.responseStart=yx(e.responseStart),e.secureConnectionStart=yx(e.secureConnectionStart),e.domainLookupStart=yx(e.domainLookupStart),e.isPerformanceNavigationTiming=!0,a=e):a=X.timing}else a=T("csi_performance_timing_to_object")?JSON.parse(JSON.stringify(X.timing)):X.timing;return a}
function yx(a){return Math.round(zx()+a)}
function zx(){return(T("csi_use_time_origin")||T("csi_use_time_origin_tvhtml5"))&&X.timeOrigin?Math.floor(X.timeOrigin):X.timing.navigationStart}
var X=vx.performance||vx.mozPerformance||vx.msPerformance||vx.webkitPerformance||new wx;var Ax=!1,Bx=!1,Cx={'script[name="scheduler/scheduler"]':"sj",'script[name="player/base"]':"pj",'link[rel="preload"][name="player/embed"]':"pej",'link[rel="stylesheet"][name="www-player"]':"pc",'link[rel="stylesheet"][name="player/www-player"]':"pc",'script[name="desktop_polymer/desktop_polymer"]':"dpj",'link[rel="import"][name="desktop_polymer"]':"dph",'script[name="mobile-c3"]':"mcj",'link[rel="stylesheet"][name="mobile-c3"]':"mcc",'script[name="player-plasma-ias-phone/base"]':"mcppj",'script[name="player-plasma-ias-tablet/base"]':"mcptj",
'link[rel="stylesheet"][name="mobile-polymer-player-ias"]':"mcpc",'link[rel="stylesheet"][name="mobile-polymer-player-svg-ias"]':"mcpsc",'script[name="mobile_blazer_core_mod"]':"mbcj",'link[rel="stylesheet"][name="mobile_blazer_css"]':"mbc",'script[name="mobile_blazer_logged_in_users_mod"]':"mbliuj",'script[name="mobile_blazer_logged_out_users_mod"]':"mblouj",'script[name="mobile_blazer_noncore_mod"]':"mbnj","#player_css":"mbpc",'script[name="mobile_blazer_desktopplayer_mod"]':"mbpj",'link[rel="stylesheet"][name="mobile_blazer_tablet_css"]':"mbtc",
'script[name="mobile_blazer_watch_mod"]':"mbwj",'script[name="embed_client"]':"ecj",'link[rel="stylesheet"][name="embed-ui"]':"ecc"};Ua(X.clearResourceTimings||X.webkitClearResourceTimings||X.mozClearResourceTimings||X.msClearResourceTimings||X.oClearResourceTimings||ui,X);function Dx(a,b){if(!T("web_csi_action_sampling_enabled")||!fx(b).actionDisabled){var c=nx(b||"");gw(c.info,a);a.loadType&&(c=a.loadType,hx(b).loadType=c);gw(kx(b),a);c=lx(b);b=fx(b).cttAuthInfo;tx().info(a,c,b)}}
function Ex(){var a,b,c,d;return((d=Dt().resolve(wt(Nq))==null?void 0:(a=Oq())==null?void 0:(b=a.loggingHotConfig)==null?void 0:(c=b.csiConfig)==null?void 0:c.debugTicks)!=null?d:[]).map(function(e){return Object.values(e)[0]})}
function Fx(a,b,c){if(!T("web_csi_action_sampling_enabled")||!fx(c).actionDisabled){var d=lx(c),e;if(e=T("web_csi_debug_sample_enabled")&&d){(Dt().resolve(wt(Nq))==null?0:Oq())&&!Bx&&(Bx=!0,Fx("gcfl",V(),c));var f,g,h;e=(Dt().resolve(wt(Nq))==null?void 0:(f=Oq())==null?void 0:(g=f.loggingHotConfig)==null?void 0:(h=g.csiConfig)==null?void 0:h.debugSampleWeight)||0;if(f=e!==0)b:{f=Ex();if(f.length>0)for(g=0;g<f.length;g++)if(a===f[g]){f=!0;break b}f=!1}if(f){for(g=f=0;g<d.length;g++)f=f*31+d.charCodeAt(g),
g<d.length-1&&(f%=0x800000000000);e=f%1E5%e!==0;fx(c).debugTicksExcludedLogged||(f={},f.debugTicksExcluded=e,Dx(f,c));fx(c).debugTicksExcludedLogged=!0}else e=!1}if(!e){if(a[0]!=="_"&&(e=a,f=b,X.mark))if(e.startsWith("mark_")||(e="mark_"+e),c&&(e+=" ("+c+")"),f===void 0||T("web_csi_disable_alt_time_performance_mark"))X.mark(e);else{f=T("csi_use_performance_navigation_timing")||T("csi_use_performance_navigation_timing_tvhtml5")?f-X.timeOrigin:f-(X.timeOrigin||X.timing.navigationStart);try{X.mark(e,
{startTime:f})}catch(k){}}e=nx(c||"");e.tick[a]=b||V();if(e.callback&&e.callback[a])for(e=y(e.callback[a]),f=e.next();!f.done;f=e.next())f=f.value,f();e=jx(c);e.gelTicks&&(e.gelTicks[a]=!0);f=ix(c);e=b||V();T("log_repeated_ytcsi_ticks")?a in f||(f[a]=e):f[a]=e;f=fx(c).cttAuthInfo;a==="_start"?(a=tx(),ux(a,"baseline_"+d)||dp("latencyActionBaselined",{clientActionNonce:d},{timestamp:b,cttAuthInfo:f})):tx().tick(a,d,b,f);Gx(c);return e}}}
function Hx(){var a=document;if("visibilityState"in a)a=a.visibilityState;else{var b=Us+"VisibilityState";a=b in a?a[b]:void 0}switch(a){case "hidden":return 0;case "visible":return 1;case "prerender":return 2;case "unloaded":return 3;default:return-1}}
function Ix(){function a(f,g,h){g=g.match("_rid")?g.split("_rid")[0]:g;typeof h==="number"&&(h=JSON.stringify(h));f.requestIds?f.requestIds.push({endpoint:g,id:h}):f.requestIds=[{endpoint:g,id:h}]}
for(var b={},c=y(Object.entries(R("TIMING_INFO",{}))),d=c.next();!d.done;d=c.next()){var e=y(d.value);d=e.next().value;e=e.next().value;switch(d){case "GetBrowse_rid":a(b,d,e);break;case "GetGuide_rid":a(b,d,e);break;case "GetHome_rid":a(b,d,e);break;case "GetPlayer_rid":a(b,d,e);break;case "GetSearch_rid":a(b,d,e);break;case "GetSettings_rid":a(b,d,e);break;case "GetTrending_rid":a(b,d,e);break;case "GetWatchNext_rid":a(b,d,e);break;case "yt_red":b.isRedSubscriber=!!e;break;case "yt_ad":b.isMonetized=
!!e}}return b}
function Jx(a,b){a=document.querySelector(a);if(!a)return!1;var c="",d=a.nodeName;d==="SCRIPT"?(c=a.src,c||(c=a.getAttribute("data-timing-href"))&&(c=window.location.protocol+c)):d==="LINK"&&(c=a.href);Gb(document)&&a.setAttribute("nonce",Gb(document));return c?(a=X.getEntriesByName(c))&&a[0]&&(a=a[0],c=zx(),Fx("rsf_"+b,c+Math.round(a.fetchStart)),Fx("rse_"+b,c+Math.round(a.responseEnd)),a.transferSize!==void 0&&a.transferSize===0)?!0:!1:!1}
function Kx(){var a=window.location.protocol,b=X.getEntriesByType("resource");b=Ub(b,function(c){return c.name.indexOf(a+"//fonts.gstatic.com/s/")===0});
(b=Wb(b,function(c,d){return d.duration>c.duration?d:c},{duration:0}))&&b.startTime>0&&b.responseEnd>0&&(Fx("wffs",yx(b.startTime)),Fx("wffe",yx(b.responseEnd)))}
function Lx(a){var b=Mx("aft",a);if(b)return b;b=R((a||"")+"TIMING_AFT_KEYS",["ol"]);for(var c=b.length,d=0;d<c;d++){var e=Mx(b[d],a);if(e)return e}return NaN}
function Mx(a,b){if(a=ix(b)[a])return typeof a==="number"?a:a[a.length-1]}
function Gx(a){var b=Mx("_start",a),c=Lx(a),d=!Ax;b&&c&&d&&(fr(qx,new px(Math.round(c-b),a)),Ax=!0)}
function Nx(){if(X.getEntriesByType){var a=X.getEntriesByType("paint");if(a=Xb(a,function(c){return c.name==="first-paint"}))return yx(a.startTime)}var b;
T("csi_use_performance_navigation_timing")||T("csi_use_performance_navigation_timing_tvhtml5")?b=X.getEntriesByType("first-paint")[0].startTime:b=X.timing.Xh;return b?Math.max(0,b):0}
;function Ox(a,b){Pm(function(){nx("").info.actionType=a;b&&Lm("TIMING_AFT_KEYS",b);Lm("TIMING_ACTION",a);var c=Ix();Object.keys(c).length>0&&Dx(c);c={isNavigation:!0,actionType:ox[R("TIMING_ACTION")]||"LATENCY_ACTION_UNKNOWN"};var d=R("PREVIOUS_ACTION");d&&(c.previousAction=ox[d]||"LATENCY_ACTION_UNKNOWN");if(d=R("CLIENT_PROTOCOL"))c.httpProtocol=d;if(d=R("CLIENT_TRANSPORT"))c.transportProtocol=d;(d=kv())&&d!=="UNDEFINED_CSN"&&(c.clientScreenNonce=d);d=Hx();if(d===1||d===-1)c.isVisible=!0;hx();gx();
c.loadType="cold";d=gx();var e=xx(),f=zx(),g=R("CSI_START_TIMESTAMP_MILLIS",0);g>0&&!T("embeds_web_enable_csi_start_override_killswitch")&&(f=g);f&&(Fx("srt",e.responseStart),d.prerender!==1&&Fx("_start",f,void 0));d=Nx();d>0&&Fx("fpt",d);d=xx();d.isPerformanceNavigationTiming&&Dx({performanceNavigationTiming:!0},void 0);Fx("nreqs",d.requestStart,void 0);Fx("nress",d.responseStart,void 0);Fx("nrese",d.responseEnd,void 0);d.redirectEnd-d.redirectStart>0&&(Fx("nrs",d.redirectStart,void 0),Fx("nre",
d.redirectEnd,void 0));d.domainLookupEnd-d.domainLookupStart>0&&(Fx("ndnss",d.domainLookupStart,void 0),Fx("ndnse",d.domainLookupEnd,void 0));d.connectEnd-d.connectStart>0&&(Fx("ntcps",d.connectStart,void 0),Fx("ntcpe",d.connectEnd,void 0));d.secureConnectionStart>=zx()&&d.connectEnd-d.secureConnectionStart>0&&(Fx("nstcps",d.secureConnectionStart,void 0),Fx("ntcpe",d.connectEnd,void 0));X&&"getEntriesByType"in X&&Kx();d=[];if(document.querySelector&&X&&X.getEntriesByName)for(var h in Cx)Cx.hasOwnProperty(h)&&
(e=Cx[h],Jx(h,e)&&d.push(e));if(d.length>0)for(c.resourceInfo=[],h=y(d),d=h.next();!d.done;d=h.next())c.resourceInfo.push({resourceCache:d.value});Dx(c);c=jx();c.preLoggedGelInfos||(c.preLoggedGelInfos=[]);h=c.preLoggedGelInfos;c=kx();d=void 0;for(e=0;e<h.length;e++)if(f=h[e],f.loadType){d=f.loadType;break}if(hx().loadType==="cold"&&(c.loadType==="cold"||d==="cold")){d=ix();e=jx();e=e.gelTicks?e.gelTicks:e.gelTicks={};for(var k in d)if(!(k in e))if(typeof d[k]==="number")Fx(k,Mx(k));else if(T("log_repeated_ytcsi_ticks"))for(f=
y(d[k]),g=f.next();!g.done;g=f.next())g=g.value,Fx(k.slice(1),g);k={};d=!1;h=y(h);for(e=h.next();!e.done;e=h.next())d=e.value,gw(c,d),gw(k,d),d=!0;d&&Dx(k)}E("ytglobal.timingready_",!0);k=R("TIMING_ACTION");F("ytglobal.timingready_")&&k&&Px()&&Lx()&&Gx()})()}
function Px(a){return Pm(function(){return Qx("_start",a)})()}
function Rx(a,b,c){Pm(Dx)(a,b,c===void 0?!1:c)}
function Sx(a,b,c){return Pm(Fx)(a,b,c)}
function Qx(a,b){return Pm(function(){var c=ix(b);return a in c})()}
function Tx(a){if(!T("universal_csi_network_ticks"))return"";a=lc(a)||"";for(var b=Object.keys(Yq),c=0;c<b.length;c++){var d=b[c];if(a.includes(d))return d}return""}
function Ux(a){if(!T("universal_csi_network_ticks"))return function(){};
var b=Yq[a];return b?(Vx(b),function(){var c=T("universal_csi_network_ticks")?(c=Zq[a])?Vx(c):!1:!1;return c}):function(){}}
function Vx(a){return Pm(function(){if(Qx(a))return!1;Sx(a,void 0,void 0);return!0})()}
function Wx(a){Pm(function(){if(!Px("attestation_challenge_fetch")||Qx(a,"attestation_challenge_fetch"))return!1;Sx(a,void 0,"attestation_challenge_fetch");return!0})()}
function Xx(){Pm(function(){var a=lx();requestAnimationFrame(function(){setTimeout(function(){a===lx()&&Sx("ol",void 0,void 0)},0)})})()}
var Yx=window;Yx.ytcsi&&(Yx.ytcsi.infoGel=Rx,Yx.ytcsi.tick=Sx);var Zx="tokens consistency mss client_location entities adblock_detection response_received_commands store PLAYER_PRELOAD shorts_prefetch".split(" "),$x=["type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.BrowseResponse","type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.PlayerResponse"];function ay(a,b,c,d){this.u=a;this.fa=b;this.j=c;this.o=d;this.i=void 0;this.h=new Map;a.cc||(a.cc={});a.cc=Object.assign({},ex,a.cc)}
function by(a,b,c,d){if(ay.instance!==void 0){if(d=ay.instance,a=[a!==d.u,b!==d.fa,c!==d.j,!1,!1,!1,void 0!==d.i],a.some(function(e){return e}))throw new U("InnerTubeTransportService is already initialized",a);
}else ay.instance=new ay(a,b,c,d)}
function cy(a){var b={signalServiceEndpoint:{signal:"GET_DATASYNC_IDS"}};var c=c===void 0?Un:c;var d=dy(a,b);return d?new wi(function(e,f){var g,h,k,l,m;return B(function(n){switch(n.h){case 1:return n.yield(d,2);case 2:g=n.i;h=g.u(b,void 0,c);if(!h){f(new U("Error: Failed to build request for command.",b));n.B(0);break}Hw(h.input);l=((k=h.ab)==null?void 0:k.mode)==="cors"?"cors":void 0;if(a.j.Nd){m=ey(h.config,l);n.B(4);break}return n.yield(fy(h.config,l),5);case 5:m=n.i;case 4:e(gy(a,h,m)),n.h=
0}})}):Bi(new U("Error: No request builder found for command.",b))}
function hy(a,b){function c(){}
var d="/youtubei/v1/"+$w(Nv);var e=e===void 0?{Pb:{identity:Un}}:e;var f=f===void 0?!0:f;c=Ux(Tx(d));b.context||(b.context=Tw(void 0,f));return new wi(function(g){var h,k,l,m,n;return B(function(r){if(r.h==1)return h=uv(d),k=bn(h)?"same-origin":"cors",a.j.Nd?(l=ey(e,k),r.B(2)):r.yield(fy(e,k),3);r.h!=2&&(l=r.i);m=vv(uv(d));n={input:m,ab:wv(m),Ga:b,config:e};g(gy(a,n,l,c));r.h=0})})}
function iy(a,b,c){var d;if(b&&!(b==null?0:(d=b.sequenceMetaData)==null?0:d.skipProcessing)&&a.o){d=y(Zx);for(var e=d.next();!e.done;e=d.next())e=e.value,a.o[e]&&a.o[e].handleResponse(b,c)}}
function gy(a,b,c,d){d=d===void 0?function(){}:d;
var e,f,g,h,k,l,m,n,r,t,w,x,z,G,H,S,Z,mb,Rb,Wa,Cb,Xa,Na,Ja,Ia,th,Ns,Os,Ps,Qs;return B(function(ha){switch(ha.h){case 1:ha.B(2);break;case 3:if((e=ha.i)&&!e.isExpired())return ha.return(Promise.resolve(e.h()));case 2:if(!((f=b)==null?0:(g=f.Ga)==null?0:g.context)){ha.B(4);break}h=b.Ga.context;ha.B(5);break;case 5:k=y([]),l=k.next();case 8:if(l.done){ha.B(4);break}m=l.value;return ha.yield(m.Yh(h),9);case 9:l=k.next();ha.B(8);break;case 4:if((n=a.i)==null||!n.hi(b.input,b.Ga)){ha.B(12);break}return ha.yield(a.i.Sh(b.input,
b.Ga),13);case 13:return r=ha.i,iy(a,r,b),ha.return(r);case 12:return(x=(w=b.config)==null?void 0:w.bi)&&a.h.has(x)?t=a.h.get(x):(z=JSON.stringify(b.Ga),S=(H=(G=b.ab)==null?void 0:G.headers)!=null?H:{},b.ab=Object.assign({},b.ab,{headers:Object.assign({},S,c)}),Z=Object.assign({},b.ab),b.ab.method==="POST"&&(Z=Object.assign({},Z,{body:z})),((mb=b.config)==null?0:mb.Ye)&&Sx(b.config.Ye),Rb=function(){return a.fa.fetch(b.input,Z,b.config)},t=Rb(),x&&a.h.set(x,t)),ha.yield(t,14);
case 14:if((Wa=ha.i)&&"error"in Wa&&((Cb=Wa)==null?0:(Xa=Cb.error)==null?0:Xa.details))for(Na=Wa.error.details,Ja=y(Na),Ia=Ja.next();!Ia.done;Ia=Ja.next())th=Ia.value,(Ns=th["@type"])&&$x.indexOf(Ns)>-1&&(delete th["@type"],Wa=th);x&&a.h.has(x)&&a.h.delete(x);((Os=b.config)==null?0:Os.Ze)&&Sx(b.config.Ze);if(Wa||(Ps=a.i)==null||!Ps.xh(b.input,b.Ga)){ha.B(15);break}return ha.yield(a.i.Rh(b.input,b.Ga),16);case 16:Wa=ha.i;case 15:return iy(a,Wa,b),((Qs=b.config)==null?0:Qs.Ve)&&Sx(b.config.Ve),d(),
ha.return(Wa||void 0)}})}
function dy(a,b){a:{a=a.u;var c,d=(c=bu(b,qm))==null?void 0:c.signal;if(d&&a.cc&&(c=a.cc[d])){var e=c();break a}var f;if((c=(f=bu(b,om))==null?void 0:f.request)&&a.he&&(f=a.he[c])){e=f();break a}for(e in b)if(a.Kc[e]&&(b=a.Kc[e])){e=b();break a}e=void 0}if(e!==void 0)return Promise.resolve(e)}
function fy(a,b){var c,d,e,f;return B(function(g){if(g.h==1){e=(c=a)==null?void 0:(d=c.Pb)==null?void 0:d.sessionIndex;var h=g.yield;var k=Tn(0,{sessionIndex:e});if(!(k instanceof wi)){var l=new wi(ui);xi(l,2,k);k=l}return h.call(g,k,2)}f=g.i;return g.return(Promise.resolve(Object.assign({},Uw(b),f)))})}
function ey(a,b){var c;a=a==null?void 0:(c=a.Pb)==null?void 0:c.sessionIndex;c=Tn(0,{sessionIndex:a});return Object.assign({},Uw(b),c)}
;var jy=new ut("INNERTUBE_TRANSPORT_TOKEN");function ky(){}
v(ky,bx);ky.prototype.j=function(){return Tv};
ky.prototype.i=function(a){return bu(a,Cm)||void 0};
ky.prototype.h=function(a,b,c){c=c===void 0?{}:c;b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params);c.botguardResponse&&(a.botguardResponse=c.botguardResponse);c.feature&&(a.clientFeature=c.feature)};
da.Object.defineProperties(ky.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});function ly(){}
v(ly,bx);ly.prototype.j=function(){return Uv};
ly.prototype.i=function(a){return bu(a,Bm)||void 0};
ly.prototype.h=function(a,b){b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params)};
da.Object.defineProperties(ly.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});var my=new ut("SHARE_CLIENT_PARAMS_PROVIDER_TOKEN");function ny(a){this.H=a}
v(ny,bx);ny.prototype.j=function(){return Ov};
ny.prototype.i=function(a){return bu(a,um)||bu(a,wm)||bu(a,tm)};
ny.prototype.h=function(a,b){b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);if(b.clientParamIdentifier){var c;if((c=this.H)==null?0:c.h(b.clientParamIdentifier))a.clientParams=this.H.i(b.clientParamIdentifier)}};
ny[tt]=[my];function oy(){}
v(oy,bx);oy.prototype.j=function(){return Qv};
oy.prototype.i=function(a){return bu(a,sm)||void 0};
oy.prototype.h=function(a,b,c){a.feedbackTokens=[];b.feedbackToken&&a.feedbackTokens.push(b.feedbackToken);if(b=b.cpn||c.cpn)a.feedbackContext={cpn:b};a.isFeedbackTokenUnencrypted=!!c.is_feedback_token_unencrypted;a.shouldMerge=!1;c.extra_feedback_tokens&&(a.shouldMerge=!0,a.feedbackTokens=a.feedbackTokens.concat(c.extra_feedback_tokens))};
da.Object.defineProperties(oy.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});function py(){}
v(py,bx);py.prototype.j=function(){return Rv};
py.prototype.i=function(a){return bu(a,zm)||void 0};
py.prototype.h=function(a,b){b.params&&(a.params=b.params);b.secondaryParams&&(a.secondaryParams=b.secondaryParams)};function qy(){}
v(qy,bx);qy.prototype.j=function(){return Sv};
qy.prototype.i=function(a){return bu(a,ym)||void 0};
qy.prototype.h=function(a,b){b.actions&&(a.actions=b.actions);b.params&&(a.params=b.params);b.playlistId&&(a.playlistId=b.playlistId)};function ry(){}
v(ry,bx);ry.prototype.j=function(){return Pv};
ry.prototype.i=function(a){return bu(a,xm)};
ry.prototype.h=function(a,b,c){c=c===void 0?{}:c;b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);c.includeListId&&(a.includeListId=!0)};var sy=new ut("FETCH_FN_TOKEN"),ty=new ut("PARSE_FN_TOKEN"),uy=new ut("WINDOW_REQUEST_TOKEN");function vy(a,b){var c=C.apply(2,arguments);a=a===void 0?0:a;U.call(this,b,c);this.errorType=a;Object.setPrototypeOf(this,this.constructor.prototype)}
v(vy,U);var wy=new ut("NETWORK_SLI_TOKEN");function xy(a,b,c,d){this.h=a;this.i=b;this.j=c;this.o=d}
xy.prototype.fetch=function(a,b,c){var d=this,e,f,g;return B(function(h){e=yy(d,a,b);g=(f=d.i)!=null?f:fetch;return h.return(g(e).then(function(k){return d.handleResponse(k,c)}).catch(function(k){Tu(k);
if((c==null?0:c.re)&&k instanceof vy&&k.errorType===1)return Promise.reject(k)}))})};
function yy(a,b,c){if(a.h){var d=lc(vc(b,"key"))||"/UNKNOWN_PATH";a.h.start(d)}d=c;T("wug_networking_gzip_request")&&(d=Hr(c));var e;return new ((e=a.o)!=null?e:window.Request)(b,d)}
xy.prototype.handleResponse=function(a,b){var c,d=(c=this.j)!=null?c:JSON.parse;c=a.text().then(function(e){if((b==null?0:b.He)&&a.ok)return Pf(b.He,e);e=e.replace(")]}'","");if((b==null?0:b.re)&&e)try{var f=d(e)}catch(h){throw new vy(1,"JSON parsing failed after fetch");}var g;return(g=f)!=null?g:d(e)});
a.redirected||a.ok?this.h&&this.h.success():(this.h&&this.h.Lh(),c=c.then(function(e){Tu(new U("Error: API fetch failed",a.status,a.url,e));return Object.assign({},e,{errorMetadata:{status:a.status}})}));
return c};
xy[tt]=[wt(wy),wt(sy),wt(ty),wt(uy)];var zy=new ut("NETWORK_MANAGER_TOKEN");var Ay;function By(a){var b=new vj;if(a.interpreterJavascript){var c=gm(a.interpreterJavascript);c=Jb(c).toString();var d=new tj;Mf(d,6,c);Hf(b,tj,1,d,ge)}else a.interpreterUrl&&(c=hm(a.interpreterUrl),c=pb(c).toString(),d=new uj,Mf(d,4,c),Hf(b,uj,2,d,ge));a.interpreterHash&&Nf(b,3,a.interpreterHash);a.program&&Nf(b,4,a.program);a.globalName&&Nf(b,5,a.globalName);a.clientExperimentsStateBlob&&Nf(b,7,a.clientExperimentsStateBlob);return b}
function Cy(a){var b={};a=y(a.split("&"));for(var c=a.next();!c.done;c=a.next())c=c.value.split("="),c.length===2&&(b[c[0]]=c[1]);return b}
;function Bc(){if(T("bg_st_hr"))return"havuokmhhs-0";var a,b=((a=performance)==null?void 0:a.timeOrigin)||0;return"havuokmhhs-"+Math.floor(b)}
function Dy(a){this.h=a}
Dy.prototype.bindInnertubeChallengeFetcher=function(a){this.h.bicf(a)};
Dy.prototype.registerChallengeFetchedCallback=function(a){this.h.bcr(a)};
Dy.prototype.getLatestChallengeResponse=function(){return this.h.blc()};
function Ey(){return new Promise(function(a){var b=window.top;b.ntpevasrs!==void 0?a(new Dy(b.ntpevasrs)):(b.ntpqfbel===void 0&&(b.ntpqfbel=[]),b.ntpqfbel.push(function(c){a(new Dy(c))}))})}
;var Fy=[],Gy=!1;function Hy(){if(xv()){var a=R("PLAYER_VARS",{});if(Ag(a)!="1"&&!zv(a)){var b=function(){Gy=!0;"google_ad_status"in window?Lm("DCLKSTAT",1):Lm("DCLKSTAT",2)};
try{Xv("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}Fy.push(Xj.pa(function(){if(!(Gy||"google_ad_status"in window)){try{aw("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}Gy=!0;Lm("DCLKSTAT",3)}},5E3))}}}
function Iy(){var a=Number(R("DCLKSTAT",0));return isNaN(a)?0:a}
;function Y(a){this.h=a}
[new Y("b.f_"),new Y("j.s_"),new Y("r.s_"),new Y("e.h_"),new Y("i.s_"),new Y("s.t_"),new Y("p.h_"),new Y("s.i_"),new Y("f.i_"),new Y("a.b_"),new Y("a.o_"),new Y("g.o_"),new Y("p.i_"),new Y("p.m_"),new Y("n.k_"),new Y("i.f_"),new Y("a.s_"),new Y("m.c_"),new Y("n.h_"),new Y("o.p_"),new Y("m.p_"),new Y("o.a_"),new Y("d.p_")].reduce(function(a,b){a[b.h]=b;return a},{});function Jy(a,b,c){var d=this;this.network=a;this.options=b;this.o=c;this.h=null;if(b.mi){var e=new zj;this.h=e.promise;D.ytAtRC&&Xj.Sa(function(){var f,g;return B(function(h){if(h.h==1){if(!D.ytAtRC)return h.return();f=Ky(null);return h.yield(d.ib(f),2)}g=h.i;D.ytAtRC&&D.ytAtRC(JSON.stringify(g));h.h=0})},2);
Ey().then(function(f){var g,h,k,l;return B(function(m){if(m.h==1)return f.bindInnertubeChallengeFetcher(function(n){return d.ib(Ky(n))}),m.yield(Ac(),2);
g=m.i;h=f.getLatestChallengeResponse();k=h.challenge;if(!k)throw Error("BGE_MACIL");l={challenge:k,gb:Cy(k),vm:g,bgChallenge:new vj};e.resolve(l);f.registerChallengeFetchedCallback(function(n){n=n.challenge;if(!n)throw Error("BGE_MACR");n={challenge:n,gb:Cy(n),vm:g,bgChallenge:new vj};d.h=Promise.resolve(n)});
m.h=0})})}else b.preload&&Ly(this,new Promise(function(f){po(function(){f(My(d))},0)}))}
Jy.prototype.j=function(){var a=this;return B(function(b){return b.h==1?b.yield(Promise.race([a.h,null]),2):b.return(!!b.i)})};
Jy.prototype.i=function(a,b,c){var d=this,e,f,g;return B(function(h){d.h===null&&Ly(d,My(d));e=!1;f={};g=function(){var k,l,m;return B(function(n){switch(n.h){case 1:return n.yield(d.h,2);case 2:k=n.i;f.challenge=k.challenge;if(!k.vm){"c1a"in k.gb&&(f.error="ATTESTATION_ERROR_VM_NOT_INITIALIZED");n.B(3);break}l=Object.assign({},{c:k.challenge,e:a},b);va(n,4);e=!0;if(T("attbs")&&!T("attmusi")){m=k.vm.hd({wb:l});n.B(6);break}return n.yield(k.vm.snapshot({wb:l}),7);case 7:m=n.i;case 6:m?f.webResponse=
m:f.error="ATTESTATION_ERROR_VM_NO_RESPONSE";wa(n,3);break;case 4:xa(n),f.error="ATTESTATION_ERROR_VM_INTERNAL_ERROR";case 3:if(a==="ENGAGEMENT_TYPE_PLAYBACK"){var r=k.gb,t={};r.c6a&&(t.reportingStatus=String(Number(r.c)^Iy()));r.c6b&&(t.broadSpectrumDetectionResult=String(Number(r.c)^Number(R("CATSTAT",0))));f.adblockReporting=t}return n.return(f)}})};
return h.return(Promise.race([g(),Ny(c,function(){var k=Object.assign({},f);e&&(k.error="ATTESTATION_ERROR_VM_TIMEOUT");return k})]))})};
function Ky(a){var b={engagementType:"ENGAGEMENT_TYPE_UNBOUND"};a&&(b.interpreterHash=a);return b}
function My(a,b){b=b===void 0?0:b;var c,d,e,f,g,h,k,l,m,n,r,t;return B(function(w){switch(w.h){case 1:c=Ky(Ej().h);if(T("att_fet_ks"))return va(w,7),w.yield(a.ib(c),9);va(w,4);return w.yield(Oy(a,c),6);case 6:g=w.i;e=g.Qe;f=g.Re;d=g;wa(w,3);break;case 4:return xa(w),Tu(Error("Failed to fetch attestation challenge after "+(b+" attempts; not retrying for 24h."))),Py(a,864E5),w.return({challenge:"",gb:{},vm:void 0,bgChallenge:void 0});case 9:d=w.i;if(!d)throw Error("Fetching Attestation challenge returned falsy");
if(!d.challenge)throw Error("Missing Attestation challenge");e=d.challenge;f=Cy(e);if("c1a"in f&&(!d.bgChallenge||!d.bgChallenge.program))throw Error("Expected bg challenge but missing.");wa(w,3);break;case 7:h=xa(w);Tu(h);b++;if(b>=5)return Tu(Error("Failed to fetch attestation challenge after "+(b+" attempts; not retrying for 24h."))),Py(a,864E5),w.return({challenge:"",gb:{},vm:void 0,bgChallenge:void 0});k=1E3*Math.pow(2,b-1)+Math.random()*1E3;return w.return(new Promise(function(x){po(function(){x(My(a,
b))},k)}));
case 3:l=Number(f.t)||7200;Py(a,l*1E3);m=void 0;if(!("c1a"in f&&d.bgChallenge)){w.B(10);break}n=By(d.bgChallenge);va(w,11);return w.yield(Fj(Ej(),n),13);case 13:wa(w,12);break;case 11:return r=xa(w),Tu(r),w.return({challenge:e,gb:f,vm:m,bgChallenge:n});case 12:return va(w,14),m=new Bj({challenge:n,Bd:{Da:"aGIf"}}),w.yield(m.cd,16);case 16:wa(w,10);break;case 14:t=xa(w),Tu(t),m=void 0;case 10:return w.return({challenge:e,gb:f,vm:m,bgChallenge:n})}})}
Jy.prototype.ib=function(a){var b=this,c;return B(function(d){c=b.o;if(!c||c.ta())return d.return(b.network.ib(a));Wx("att_pna");return d.return(new Promise(function(e){bi(c,"publicytnetworkstatus-online",function(){b.network.ib(a).then(e)})}))})};
function Qy(a){if(!a)throw Error("Fetching Attestation challenge returned falsy");if(!a.challenge)throw Error("Missing Attestation challenge");var b=a.challenge,c=Cy(b);if("c1a"in c&&(!a.bgChallenge||!a.bgChallenge.program))throw Error("Expected bg challenge but missing.");return Object.assign({},a,{Qe:b,Re:c})}
function Oy(a,b){var c,d,e,f,g;return B(function(h){switch(h.h){case 1:c=void 0,d=0,e={};case 2:if(!(d<5)){h.B(4);break}if(!(d>0)){h.B(5);break}e.pd=1E3*Math.pow(2,d-1)+Math.random()*1E3;return h.yield(new Promise(function(k){return function(l){po(function(){l(void 0)},k.pd)}}(e)),5);
case 5:return va(h,7),h.yield(a.ib(b),9);case 9:return f=h.i,h.return(Qy(f));case 7:c=g=xa(h),g instanceof Error&&Tu(g);case 8:d++;e={pd:void 0};h.B(2);break;case 4:throw c;}})}
function Ly(a,b){a.h=b}
function Ry(a){var b,c,d;return B(function(e){if(e.h==1)return e.yield(Promise.race([a.h,null]),2);b=e.i;var f=My(a);a.h=f;(c=b)==null||(d=c.vm)==null||d.dispose();e.h=0})}
function Py(a,b){function c(){var e;return B(function(f){e=d-Date.now();return e<1E3?f.yield(Ry(a),0):(po(c,Math.min(e,6E4)),f.B(0))})}
var d=Date.now()+b;c()}
function Ny(a,b){return new Promise(function(c){po(function(){c(b())},a)})}
;function Sy(a){this.h=a}
Sy.prototype.ib=function(a){Wx("att_fsr");return hy(this.h,a).then(function(b){Wx("att_frr");return b})};function Ty(){var a,b,c;return B(function(d){if(d.h==1)return a=Dt().resolve(jy),a?d.yield(cy(a),2):(Tu(Error("InnertubeTransportService unavailable in fetchDatasyncIds")),d.return(void 0));if(b=d.i){if(b.errorMetadata)return Tu(Error("Datasync IDs fetch responded with "+b.errorMetadata.status+": "+b.error)),d.return(void 0);c=b.Ah;return d.return(c)}Tu(Error("Network request to get Datasync IDs failed."));return d.return(void 0)})}
;function Uy(){}
v(Uy,bx);Uy.prototype.j=function(){return Qv};
Uy.prototype.i=function(a){return bu(a,Am)};
Uy.prototype.h=function(a,b){b.undoToken&&(a.feedbackTokens=[b.undoToken]);b.isUndoTokenUnencrypted&&(a.isFeedbackTokenUnencrypted=b.isUndoTokenUnencrypted)};
da.Object.defineProperties(Uy.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});function Vy(){var a;return(a=R("WEB_PLAYER_CONTEXT_CONFIGS"))==null?void 0:a.WEB_PLAYER_CONTEXT_CONFIG_ID_EMBEDDED_PLAYER}
;var Wy=D.caches,Xy;function Yy(a){var b=a.indexOf(":");return b===-1?{Ed:a}:{Ed:a.substring(0,b),datasyncId:a.substring(b+1)}}
function Zy(){return B(function(a){if(Xy!==void 0)return a.return(Xy);Xy=new Promise(function(b){var c;return B(function(d){switch(d.h){case 1:return va(d,2),d.yield(Wy.open("test-only"),4);case 4:return d.yield(Wy.delete("test-only"),5);case 5:wa(d,3);break;case 2:if(c=xa(d),c instanceof Error&&c.name==="SecurityError")return b(!1),d.return();case 3:b("caches"in window),d.h=0}})});
return a.return(Xy)})}
function $y(a){var b,c,d,e,f,g,h;B(function(k){if(k.h==1)return k.yield(Zy(),2);if(k.h!=3){if(!k.i)return k.return(!1);b=[];return k.yield(Wy.keys(),3)}c=k.i;d=y(c);for(e=d.next();!e.done;e=d.next())f=e.value,g=Yy(f),h=g.datasyncId,!h||a.includes(h)||b.push(Wy.delete(f));return k.return(Promise.all(b).then(function(l){return l.some(function(m){return m})}))})}
function az(){var a,b,c,d,e,f,g;return B(function(h){if(h.h==1)return h.yield(Zy(),2);if(h.h!=3){if(!h.i)return h.return(!1);a=no("cache contains other");return h.yield(Wy.keys(),3)}b=h.i;c=y(b);for(d=c.next();!d.done;d=c.next())if(e=d.value,f=Yy(e),(g=f.datasyncId)&&g!==a)return h.return(!0);return h.return(!1)})}
;function bz(){try{return!!self.sessionStorage}catch(a){return!1}}
;function cz(a){a=a.match(/(.*)::.*::.*/);if(a!==null)return a[1]}
function dz(a){if(bz()){var b=Object.keys(window.sessionStorage);b=y(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=cz(c);d===void 0||a.includes(d)||self.sessionStorage.removeItem(c)}}}
function ez(){if(!bz())return!1;var a=no(),b=Object.keys(window.sessionStorage);b=y(b);for(var c=b.next();!c.done;c=b.next())if(c=cz(c.value),c!==void 0&&c!==a)return!0;return!1}
;function fz(){Ty().then(function(a){a&&(sq(a),$y(a),Cw(a),dz(a))})}
function gz(){var a=new Bs;Xj.pa(function(){var b,c,d,e,f;return B(function(g){switch(g.h){case 1:if(T("ytidb_clear_optimizations_killswitch")){g.B(2);break}b=no("clear");if(b.startsWith("V")&&b.endsWith("||")){var h=[b];sq(h);$y(h);Cw(h);dz(h);return g.return()}c=Dw();d=ez();return g.yield(az(),3);case 3:return e=g.i,g.yield(tq(),4);case 4:if(f=g.i,!(c||d||e||f))return g.return();case 2:a.ta()?fz():bi(a,"publicytnetworkstatus-online",fz),g.h=0}})})}
;var hz=["www.youtube-nocookie.com","www.youtubeeducation.com","youtube.googleapis.com"];function iz(){this.state=1;this.vm=null;this.h=void 0}
p=iz.prototype;p.initialize=function(a,b,c,d){this.h=d;if(a.program){var e;d=(e=a.interpreterUrl)!=null?e:null;if(a.interpreterSafeScript)e=gm(a.interpreterSafeScript);else{var f;e=(f=a.interpreterScript)!=null?f:null}a.interpreterSafeUrl&&(d=hm(a.interpreterSafeUrl).toString());jz(this,e,d,a.program,b,c)}else Tu(Error("BL:CIP"))};
function jz(a,b,c,d,e,f){var g=g===void 0?"trayride":g;c?(a.state=2,Xv(c,function(){window[g]?kz(a,d,g,e):(a.state=3,Zv(c),Tu(new U("BL:ULB",""+c)))},f)):b?(f=Jg("SCRIPT"),b instanceof Hb?(f.textContent=Jb(b),Kb(f)):f.textContent=b,f.nonce=Gb(document),document.head.appendChild(f),document.head.removeChild(f),window[g]?kz(a,d,g,e):(a.state=4,Tu(new U("BL:ULBJ")))):Tu(new U("BL:ULV"))}
p.isLoading=function(){return this.state===2};
function kz(a,b,c,d){a.state=5;var e=!!a.h&&hz.includes(kc(a.h)||"");try{var f=new Bj({program:b,globalName:c,Bd:{disable:!T("att_web_record_metrics")||!T("att_skip_metrics_for_cookieless_domains_ks")&&e,Da:"aGIf"}});f.cd.then(function(){a.state=6;d&&d(b)});
a.bd(f)}catch(g){a.state=7,g instanceof Error&&Tu(g)}}
p.invoke=function(a){a=a===void 0?{}:a;return this.ld()?this.Rd({wb:a}):null};
p.dispose=function(){this.bd(null);this.state=8};
p.ld=function(){return!!this.vm};
p.Rd=function(a){return this.vm.hd(a)};
p.bd=function(a){xc(this.vm);this.vm=a};function lz(){var a=F("yt.abuse.playerAttLoader");return a&&["bgvma","bgvmb","bgvmc"].every(function(b){return b in a})?a:null}
;function mz(){iz.apply(this,arguments)}
v(mz,iz);mz.prototype.bd=function(a){var b;(b=lz())==null||b.bgvma();a?(b={bgvma:a.dispose.bind(a),bgvmb:a.snapshot.bind(a),bgvmc:a.hd.bind(a)},E("yt.abuse.playerAttLoader",b),E("yt.abuse.playerAttLoaderRun",function(c){return a.snapshot(c)})):(E("yt.abuse.playerAttLoader",null),E("yt.abuse.playerAttLoaderRun",null))};
mz.prototype.ld=function(){return!!lz()};
mz.prototype.Rd=function(a){return lz().bgvmc(a)};var nz=new ut("AUTH_SERVICE_TOKEN");function oz(a){Nt.call(this,a===void 0?"document_active":a);var b=this;this.o=10;this.h=new Map;this.transitions=[{from:"document_active",to:"document_disposed_preventable",action:this.G},{from:"document_active",to:"document_disposed",action:this.u},{from:"document_disposed_preventable",to:"document_disposed",action:this.u},{from:"document_disposed_preventable",to:"flush_logs",action:this.H},{from:"document_disposed_preventable",to:"document_active",action:this.i},{from:"document_disposed",to:"flush_logs",
action:this.H},{from:"document_disposed",to:"document_active",action:this.i},{from:"document_disposed",to:"document_disposed",action:function(){}},
{from:"flush_logs",to:"document_active",action:this.i}];window.addEventListener("pagehide",function(c){b.transition("document_disposed",{event:c})});
window.addEventListener("beforeunload",function(c){b.transition("document_disposed_preventable",{event:c})})}
v(oz,Nt);oz.prototype.G=function(a,b){if(!this.h.get("document_disposed_preventable")){a(b==null?void 0:b.event);var c,d;if((b==null?0:(c=b.event)==null?0:c.defaultPrevented)||(b==null?0:(d=b.event)==null?0:d.returnValue)){b.event.returnValue||(b.event.returnValue=!0);b.event.defaultPrevented||b.event.preventDefault();this.h=new Map;this.transition("document_active");return}}this.h.set("document_disposed_preventable",!0);this.h.get("document_disposed")?this.transition("flush_logs"):this.transition("document_disposed")};
oz.prototype.u=function(a,b){this.h.get("document_disposed")?this.transition("document_active"):(a(b==null?void 0:b.event),this.h.set("document_disposed",!0),this.transition("flush_logs"))};
oz.prototype.H=function(a,b){a(b==null?void 0:b.event);this.transition("document_active")};
oz.prototype.i=function(){this.h=new Map};function pz(a){Nt.call(this,a===void 0?"document_visibility_unknown":a);var b=this;this.transitions=[{from:"document_visibility_unknown",to:"document_visible",action:this.i},{from:"document_visibility_unknown",to:"document_hidden",action:this.h},{from:"document_visibility_unknown",to:"document_foregrounded",action:this.H},{from:"document_visibility_unknown",to:"document_backgrounded",action:this.u},{from:"document_visible",to:"document_hidden",action:this.h},{from:"document_visible",to:"document_foregrounded",
action:this.H},{from:"document_visible",to:"document_visible",action:this.i},{from:"document_foregrounded",to:"document_visible",action:this.i},{from:"document_foregrounded",to:"document_hidden",action:this.h},{from:"document_foregrounded",to:"document_foregrounded",action:this.H},{from:"document_hidden",to:"document_visible",action:this.i},{from:"document_hidden",to:"document_backgrounded",action:this.u},{from:"document_hidden",to:"document_hidden",action:this.h},{from:"document_backgrounded",to:"document_hidden",
action:this.h},{from:"document_backgrounded",to:"document_backgrounded",action:this.u},{from:"document_backgrounded",to:"document_visible",action:this.i}];document.addEventListener("visibilitychange",function(c){document.visibilityState==="visible"?b.transition("document_visible",{event:c}):b.transition("document_hidden",{event:c})});
T("visibility_lifecycles_dynamic_backgrounding")&&(window.addEventListener("blur",function(c){b.transition("document_backgrounded",{event:c})}),window.addEventListener("focus",function(c){b.transition("document_foregrounded",{event:c})}))}
v(pz,Nt);pz.prototype.i=function(a,b){a(b==null?void 0:b.event);T("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_foregrounded")};
pz.prototype.h=function(a,b){a(b==null?void 0:b.event);T("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_backgrounded")};
pz.prototype.u=function(a,b){a(b==null?void 0:b.event)};
pz.prototype.H=function(a,b){a(b==null?void 0:b.event)};function qz(){this.o=new oz;this.u=new pz}
qz.prototype.install=function(){var a=C.apply(0,arguments),b=this;a.forEach(function(c){b.o.install(c)});
a.forEach(function(c){b.u.install(c)})};function rz(){this.o=[];this.i=new Map;this.h=new Map;this.j=new Set}
rz.prototype.clickCommand=function(a,b,c){var d=a.clickTrackingParams;c=c===void 0?0:c;if(d)if(c=kv(c===void 0?0:c)){a=this.client;d=new dv({trackingParams:d});var e=void 0;if(T("no_client_ve_attach_unless_shown")){var f=xw(d,c);tw.set(f,!0);yw(d,c)}e=e||"INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK";f=ww({cttAuthInfo:mv(c)||void 0},c);d={csn:c,ve:d.getAsJson(),gestureType:e};b&&(d.clientData=b);c==="UNDEFINED_CSN"?zw("visualElementGestured",f,d):a?Ku("visualElementGestured",d,a,f):dp("visualElementGestured",
d,f);b=!0}else b=!1;else b=!1;return b};
rz.prototype.stateChanged=function(a,b,c){this.visualElementStateChanged(new dv({trackingParams:a}),b,c===void 0?0:c)};
rz.prototype.visualElementStateChanged=function(a,b,c){c=c===void 0?0:c;if(c===0&&this.j.has(c))this.o.push([a,b]);else{var d=c;d=d===void 0?0:d;c=kv(d);a||(a=(a=hv(d===void 0?0:d))?new dv({veType:a,youtubeData:void 0,jspbYoutubeData:void 0}):null);var e=a;c&&e&&(a=this.client,d=ww({cttAuthInfo:mv(c)||void 0},c),b={csn:c,ve:e.getAsJson(),clientData:b},c==="UNDEFINED_CSN"?zw("visualElementStateChanged",d,b):a?Ku("visualElementStateChanged",b,a,d):dp("visualElementStateChanged",b,d))}};
function sz(a,b){if(b===void 0)for(var c=jv(),d=0;d<c.length;d++)c[d]!==void 0&&sz(a,c[d]);else a.i.forEach(function(e,f){(f=a.h.get(f))&&vw(a.client,b,f,e)}),a.i.clear(),a.h.clear()}
;function tz(){qz.call(this);var a={};this.install((a.document_disposed={callback:this.h},a));T("combine_ve_grafts")&&(a={},this.install((a.document_disposed={callback:this.i},a)));a={};this.install((a.flush_logs={callback:this.j},a));T("web_log_cfg_cee_ks")||po(uz)}
v(tz,qz);tz.prototype.j=function(){dp("finalPayload",{csn:kv()})};
tz.prototype.h=function(){Yu(Zu)};
tz.prototype.i=function(){var a=sz;rz.instance||(rz.instance=new rz);a(rz.instance)};
function uz(){var a=R("CLIENT_EXPERIMENT_EVENTS");if(a){var b=le();a=y(a);for(var c=a.next();!c.done;c=a.next())c=c.value,b(c)&&dp("genericClientExperimentEvent",{eventType:c});delete Km.CLIENT_EXPERIMENT_EVENTS}}
;function vz(){}
function wz(){var a=F("ytglobal.storage_");a||(a=new vz,E("ytglobal.storage_",a));return a}
vz.prototype.estimate=function(){var a,b,c;return B(function(d){a=navigator;return((b=a.storage)==null?0:b.estimate)?d.return(a.storage.estimate()):((c=a.webkitTemporaryStorage)==null?0:c.queryUsageAndQuota)?d.return(xz()):d.return()})};
function xz(){var a=navigator;return new Promise(function(b,c){var d;(d=a.webkitTemporaryStorage)!=null&&d.queryUsageAndQuota?a.webkitTemporaryStorage.queryUsageAndQuota(function(e,f){b({usage:e,quota:f})},function(e){c(e)}):c(Error("webkitTemporaryStorage is not supported."))})}
E("ytglobal.storageClass_",vz);function bp(a,b){var c=this;this.handleError=a;this.h=b;this.i=!1;self.document===void 0||self.addEventListener("beforeunload",function(){c.i=!0});
this.j=Math.random()<=.2}
bp.prototype.Ha=function(a){this.handleError(a)};
bp.prototype.logEvent=function(a,b){switch(a){case "IDB_DATA_CORRUPTED":T("idb_data_corrupted_killswitch")||this.h("idbDataCorrupted",b);break;case "IDB_UNEXPECTEDLY_CLOSED":this.h("idbUnexpectedlyClosed",b);break;case "IS_SUPPORTED_COMPLETED":T("idb_is_supported_completed_killswitch")||this.h("idbIsSupportedCompleted",b);break;case "QUOTA_EXCEEDED":yz(this,b);break;case "TRANSACTION_ENDED":this.j&&Math.random()<=.1&&this.h("idbTransactionEnded",b);break;case "TRANSACTION_UNEXPECTEDLY_ABORTED":a=
Object.assign({},b,{hasWindowUnloaded:this.i}),this.h("idbTransactionAborted",a)}};
function yz(a,b){wz().estimate().then(function(c){c=Object.assign({},b,{isSw:self.document===void 0,isIframe:self!==self.top,deviceStorageUsageMbytes:zz(c==null?void 0:c.usage),deviceStorageQuotaMbytes:zz(c==null?void 0:c.quota)});a.h("idbQuotaExceeded",c)})}
function zz(a){return typeof a==="undefined"?"-1":String(Math.ceil(a/1048576))}
;var Az={Kc:{feedbackEndpoint:Xw(oy),modifyChannelNotificationPreferenceEndpoint:Xw(py),playlistEditEndpoint:Xw(qy),shareEntityEndpoint:Xw(ny),subscribeEndpoint:Xw(ky),undoFeedbackEndpoint:Xw(Uy),unsubscribeEndpoint:Xw(ly),webPlayerShareEntityServiceEndpoint:Xw(ry)}};function Bz(){var a=Dt();zt(a,{pb:zy,Hc:xy});zt(a,{pb:nz,Hc:Rn});var b=Sw(),c=a.resolve(nz),d=a.resolve(zy),e={};b&&(e.client_location=b);by(Az,d,c,e);zt(a,{pb:jy,kd:ay.instance})}
;var Cz={},Dz=(Cz["api.invalidparam"]=2,Cz.auth=150,Cz["drm.auth"]=150,Cz["heartbeat.net"]=150,Cz["heartbeat.servererror"]=150,Cz["heartbeat.stop"]=150,Cz["html5.unsupportedads"]=5,Cz["fmt.noneavailable"]=5,Cz["fmt.decode"]=5,Cz["fmt.unplayable"]=5,Cz["html5.missingapi"]=5,Cz["html5.unsupportedlive"]=5,Cz["drm.unavailable"]=5,Cz["mrm.blocked"]=151,Cz["embedder.identity.denied"]=152,Cz);var Ez=new Set("endSeconds startSeconds mediaContentUrl suggestedQuality videoId rct rctn playmuted muted_autoplay_duration_mode".split(" "));function Fz(a){return(a.search("cue")===0||a.search("load")===0)&&a!=="loadModule"}
function Gz(a,b,c){if(typeof a==="string")return{videoId:a,startSeconds:b,suggestedQuality:c};b={};c=y(Ez);for(var d=c.next();!d.done;d=c.next())d=d.value,a[d]&&(b[d]=a[d]);return b}
function Hz(a,b,c,d){if(Oa(a)&&!Array.isArray(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};typeof a==="string"&&a.length===16?b.list="PL"+a:b.playlist=a;return b}
;function Iz(a){I.call(this);var b=this;this.api=a;this.Y=this.u=!1;this.A=[];this.P={};this.j=[];this.i=[];this.Z=!1;this.sessionId=this.h=null;this.targetOrigin="*";this.U=T("web_player_split_event_bus_iframe");this.o=R("POST_MESSAGE_ORIGIN")||document.location.protocol+"//"+document.location.hostname;this.G=function(c){a:if(!(b.o!=="*"&&c.origin!==b.o||b.h&&c.source!==b.h||typeof c.data!=="string")){try{var d=JSON.parse(c.data)}catch(h){break a}if(d)switch(d.event){case "listening":var e=c.source;
c=c.origin;d=d.id;c!=="null"&&(b.o=b.targetOrigin=c);b.h=e;b.sessionId=d;if(b.u){b.Y=!0;b.u=!1;b.sendMessage("initialDelivery",Jz(b));b.sendMessage("onReady");e=y(b.A);for(d=e.next();!d.done;d=e.next())Kz(b,d.value);b.A=[]}break;case "command":if(e=d.func,d=d.args,e==="addEventListener"&&d)e=d[0],d=c.origin,e==="onReady"?b.api.logApiCall(e+" invocation",d):e==="onError"&&b.Z&&(b.api.logApiCall(e+" invocation",d,b.errorCode),b.errorCode=void 0),b.api.logApiCall(e+" registration",d),b.P[e]||e==="onReady"||
(c=Lz(b,e,d),b.i.push({eventType:e,listener:c,origin:d}),b.U?b.api.handleExternalCall("addEventListener",[e,c],d):b.api.addEventListener(e,c),b.P[e]=!0);else if(c=c.origin,b.api.isExternalMethodAvailable(e,c)){d=d||[];if(d.length>0&&Fz(e)){var f=d;if(Oa(f[0])&&!Array.isArray(f[0]))var g=f[0];else switch(g={},e){case "loadVideoById":case "cueVideoById":g=Gz(f[0],f[1]!==void 0?Number(f[1]):void 0,f[2]);break;case "loadVideoByUrl":case "cueVideoByUrl":g=f[0];typeof g==="string"&&(g={mediaContentUrl:g,
startSeconds:f[1]!==void 0?Number(f[1]):void 0,suggestedQuality:f[2]});c:{if((f=g.mediaContentUrl)&&(f=/\/([ve]|embed)\/([^#?]+)/.exec(f))&&f[2]){f=f[2];break c}f=null}g.videoId=f;g=Gz(g);break;case "loadPlaylist":case "cuePlaylist":g=Hz(f[0],f[1],f[2],f[3])}d.length=1;d[0]=g}b.api.handleExternalCall(e,d,c);Fz(e)&&Mz(b,Jz(b))}}}};
Nz.addEventListener("message",this.G);if(a=R("WIDGET_ID"))this.sessionId=a;Oz(this,"onReady",function(){b.u=!0;var c=b.api.getVideoData();if(!c.isPlayable){b.Z=!0;c=c.errorCode;var d=d===void 0?5:d;b.errorCode=c?Dz[c]||d:d;b.sendMessage("onError",Number(b.errorCode))}});
Oz(this,"onVideoProgress",this.lf.bind(this));Oz(this,"onVolumeChange",this.mf.bind(this));Oz(this,"onApiChange",this.df.bind(this));Oz(this,"onPlaybackQualityChange",this.hf.bind(this));Oz(this,"onPlaybackRateChange",this.jf.bind(this));Oz(this,"onStateChange",this.kf.bind(this));Oz(this,"onWebglSettingsChanged",this.nf.bind(this));Oz(this,"onCaptionsTrackListChanged",this.ef.bind(this));Oz(this,"captionssettingschanged",this.ff.bind(this))}
v(Iz,I);function Mz(a,b){a.sendMessage("infoDelivery",b)}
p=Iz.prototype;p.sendMessage=function(a,b){a={event:a,info:b===void 0?null:b};this.Y?Kz(this,a):this.A.push(a)};
function Lz(a,b,c){return function(d){b==="onError"?a.api.logApiCall(b+" invocation",c,d):a.api.logApiCall(b+" invocation",c);a.sendMessage(b,d)}}
function Oz(a,b,c){a.j.push({eventType:b,listener:c});a.api.addEventListener(b,c)}
function Jz(a){if(!a.api)return null;var b=a.api.getApiInterface();Yb(b,"getVideoData");for(var c={apiInterface:b},d=0,e=b.length;d<e;d++){var f=b[d];if(f.search("get")===0||f.search("is")===0){var g=0;f.search("get")===0?g=3:f.search("is")===0&&(g=2);g=f.charAt(g).toLowerCase()+f.substring(g+1);try{var h=a.api[f]();c[g]=h}catch(k){}}}c.videoData=a.api.getVideoData();c.currentTimeLastUpdated_=Date.now()/1E3;return c}
p.kf=function(a){a={playerState:a,currentTime:this.api.getCurrentTime(),duration:this.api.getDuration(),videoData:this.api.getVideoData(),videoStartBytes:0,videoBytesTotal:this.api.getVideoBytesTotal(),videoLoadedFraction:this.api.getVideoLoadedFraction(),playbackQuality:this.api.getPlaybackQuality(),availableQualityLevels:this.api.getAvailableQualityLevels(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getVideoUrl&&
(a.videoUrl=this.api.getVideoUrl());this.api.getVideoContentRect&&(a.videoContentRect=this.api.getVideoContentRect());this.api.getProgressState&&(a.progressState=this.api.getProgressState());this.api.getPlaylist&&(a.playlist=this.api.getPlaylist());this.api.getPlaylistIndex&&(a.playlistIndex=this.api.getPlaylistIndex());this.api.getStoryboardFormat&&!T("embeds_enable_vfsyb")&&(a.storyboardFormat=this.api.getStoryboardFormat());Mz(this,a)};
p.hf=function(a){a={playbackQuality:a};this.api.getAvailableQualityLevels&&(a.availableQualityLevels=this.api.getAvailableQualityLevels());this.api.getPreferredQuality&&(a.preferredQuality=this.api.getPreferredQuality());Mz(this,a)};
p.jf=function(a){Mz(this,{playbackRate:a})};
p.df=function(){for(var a=this.api.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.api.getOptions(e);a.join(", ");b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var k=f[g],l=this.api.getOption(e,k);b[e][k]=l}}this.sendMessage("apiInfoDelivery",b)};
p.mf=function(){Mz(this,{muted:this.api.isMuted(),volume:this.api.getVolume()})};
p.lf=function(a){a={currentTime:a,videoBytesLoaded:this.api.getVideoBytesLoaded(),videoLoadedFraction:this.api.getVideoLoadedFraction(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getProgressState&&(a.progressState=this.api.getProgressState());Mz(this,a)};
p.nf=function(){Mz(this,{sphericalProperties:this.api.getSphericalProperties()})};
p.ef=function(){if(this.api.getCaptionTracks){var a={captionTracks:this.api.getCaptionTracks()};Mz(this,a)}};
p.ff=function(){if(this.api.getSubtitlesUserSettings){var a={subtitlesUserSettings:this.api.getSubtitlesUserSettings()};Mz(this,a)}};
function Kz(a,b){if(a.h){b.channel="widget";a.sessionId&&(b.id=a.sessionId);try{var c=JSON.stringify(b);a.h.postMessage(c,a.targetOrigin)}catch(d){Tu(d)}}}
p.ba=function(){I.prototype.ba.call(this);Nz.removeEventListener("message",this.G);for(var a=0;a<this.j.length;a++){var b=this.j[a];this.api.removeEventListener(b.eventType,b.listener)}this.j=[];for(a=0;a<this.i.length;a++)b=this.i[a],this.U?this.api.handleExternalCall("removeEventListener",[b.eventType,b.listener],b.origin):this.api.removeEventListener(b.eventType,b.listener);this.i=[]};
var Nz=window;function Pz(a,b,c){I.call(this);var d=this;this.api=a;this.id=b;this.origin=c;this.h={};this.j=T("web_player_split_event_bus_iframe");this.i=function(e){a:if(e.origin===d.origin){var f=e.data;if(typeof f==="string"){try{f=JSON.parse(f)}catch(k){break a}if(f.command){var g=f.command;f=f.data;e=e.origin;if(!d.ea){var h=f||{};switch(g){case "addEventListener":typeof h.event==="string"&&d.addListener(h.event,e);break;case "removeEventListener":typeof h.event==="string"&&d.removeListener(h.event,e);break;
default:d.api.isReady()&&d.api.isExternalMethodAvailable(g,e||null)&&(f=Qz(g,f||{}),f=d.api.handleExternalCall(g,f,e||null),(f=Rz(g,f))&&Sz(d,g,f))}}}}}};
Tz.addEventListener("message",this.i);Sz(this,"RECEIVING")}
v(Pz,I);p=Pz.prototype;p.addListener=function(a,b){if(!(a in this.h)){var c=this.gf.bind(this,a);this.h[a]=c;this.addEventListener(a,c,b)}};
p.gf=function(a,b){this.ea||Sz(this,a,Uz(a,b))};
p.removeListener=function(a,b){a in this.h&&(this.removeEventListener(a,this.h[a],b),delete this.h[a])};
p.addEventListener=function(a,b,c){this.j?a==="onReady"?this.api.addEventListener(a,b):this.api.handleExternalCall("addEventListener",[a,b],c||null):this.api.addEventListener(a,b)};
p.removeEventListener=function(a,b,c){this.j?a==="onReady"?this.api.removeEventListener(a,b):this.api.handleExternalCall("removeEventListener",[a,b],c||null):this.api.removeEventListener(a,b)};
function Qz(a,b){switch(a){case "loadVideoById":return[Gz(b)];case "cueVideoById":return[Gz(b)];case "loadVideoByPlayerVars":return[b];case "cueVideoByPlayerVars":return[b];case "loadPlaylist":return[Hz(b)];case "cuePlaylist":return[Hz(b)];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];case "setLoop":return[b.loopPlaylists];case "setShuffle":return[b.shufflePlaylist];
case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey,b.ctrlKey,b.altKey,b.metaKey,b.key,b.code]}return[]}
function Rz(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
function Uz(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}if(b!=null)return{value:b}}
function Sz(a,b,c){a.ea||(b={id:a.id,command:b},c&&(b.data=c),Vz.postMessage(JSON.stringify(b),a.origin))}
p.ba=function(){Tz.removeEventListener("message",this.i);for(var a in this.h)this.h.hasOwnProperty(a)&&this.removeListener(a);I.prototype.ba.call(this)};
var Tz=window,Vz=window.parent;var Wz=new mz;function Xz(){return Wz.ld()}
function Yz(a){a=a===void 0?{}:a;return Wz.invoke(a)}
;function Zz(a,b,c,d,e){I.call(this);var f=this;this.A=b;this.webPlayerContextConfig=d;this.Lb=e;this.Qa=!1;this.api={};this.ma=this.u=null;this.U=new N;this.h={};this.Z=this.xa=this.elementId=this.Ra=this.config=null;this.Y=!1;this.j=this.G=null;this.Fa={};this.Ic=["onReady"];this.lastError=null;this.fb=NaN;this.P={};this.ha=0;this.i=this.o=a;zc(this,this.U);$z(this);c?this.ha=setTimeout(function(){f.loadNewVideoConfig(c)},0):d&&(aA(this),bA(this))}
v(Zz,I);p=Zz.prototype;p.getId=function(){return this.A};
p.loadNewVideoConfig=function(a){if(!this.ea){this.ha&&(clearTimeout(this.ha),this.ha=0);var b=a||{};b instanceof Mv||(b=new Mv(b));this.config=b;this.setConfig(a);bA(this);this.isReady()&&cA(this)}};
function aA(a){var b;a.webPlayerContextConfig?b=a.webPlayerContextConfig.rootElementId:b=a.config.attrs.id;a.elementId=b||a.elementId;a.elementId==="video-player"&&(a.elementId=a.A,a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.A:a.config.attrs.id=a.A);var c;((c=a.i)==null?void 0:c.id)===a.elementId&&(a.elementId+="-player",a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.elementId:a.config.attrs.id=a.elementId)}
p.setConfig=function(a){this.Ra=a;this.config=dA(a);aA(this);if(!this.xa){var b;this.xa=eA(this,((b=this.config.args)==null?void 0:b.jsapicallback)||"onYouTubePlayerReady")}this.config.args?this.config.args.jsapicallback=null:this.config.args={jsapicallback:null};var c;if((c=this.config)==null?0:c.attrs)a=this.config.attrs,(b=a.width)&&this.i&&(this.i.style.width=Rj(Number(b)||b)),(a=a.height)&&this.i&&(this.i.style.height=Rj(Number(a)||a))};
function cA(a){if(a.config&&a.config.loaded!==!0)if(a.config.loaded=!0,!a.config.args||a.config.args.autoplay!=="0"&&a.config.args.autoplay!==0&&a.config.args.autoplay!==!1){var b;a.api.loadVideoByPlayerVars((b=a.config.args)!=null?b:null)}else a.api.cueVideoByPlayerVars(a.config.args)}
function fA(a){var b=!0,c=gA(a);c&&a.config&&(b=c.dataset.version===hA(a));return b&&!!F("yt.player.Application.create")}
function bA(a){if(!a.ea&&!a.Y){var b=fA(a);if(b&&(gA(a)?"html5":null)==="html5")a.Z="html5",a.isReady()||iA(a);else if(jA(a),a.Z="html5",b&&a.j&&a.o)a.o.appendChild(a.j),iA(a);else{a.config&&(a.config.loaded=!0);var c=!1;a.G=function(){c=!0;var d=kA(a,"player_bootstrap_method")?F("yt.player.Application.createAlternate")||F("yt.player.Application.create"):F("yt.player.Application.create");var e=a.config?dA(a.config):void 0;d&&d(a.o,e,a.webPlayerContextConfig,a.Lb);iA(a)};
a.Y=!0;b?a.G():(Xv(hA(a),a.G),(b=lA(a))&&dw(b||""),mA(a)&&!c&&E("yt.player.Application.create",null))}}}
function gA(a){var b=Ig(a.elementId);!b&&a.i&&a.i.querySelector&&(b=a.i.querySelector("#"+a.elementId));return b}
function iA(a){if(!a.ea){var b=gA(a),c=!1;b&&b.getApiInterface&&b.getApiInterface()&&(c=!0);if(c){a.Y=!1;if(!kA(a,"html5_remove_not_servable_check_killswitch")){var d;if((b==null?0:b.isNotServable)&&a.config&&(b==null?0:b.isNotServable((d=a.config.args)==null?void 0:d.video_id)))return}nA(a)}else a.fb=setTimeout(function(){iA(a)},50)}}
function nA(a){$z(a);a.Qa=!0;var b=gA(a);if(b){a.u=oA(a,b,"addEventListener");a.ma=oA(a,b,"removeEventListener");var c=b.getApiInterface();c=c.concat(b.getInternalApiInterface());for(var d=a.api,e=0;e<c.length;e++){var f=c[e];d[f]||(d[f]=oA(a,b,f))}}for(var g in a.h)a.h.hasOwnProperty(g)&&a.u&&a.u(g,a.h[g]);cA(a);a.xa&&a.xa(a.api);a.U.sb("onReady",a.api)}
function oA(a,b,c){var d=b[c];return function(){var e=C.apply(0,arguments);try{return a.lastError=null,d.apply(b,e)}catch(f){if(c!=="sendAbandonmentPing")throw f.params=c,a.lastError=f,e=new U("PlayerProxy error in method call",{error:f,method:c,playerId:a.A}),e.level="WARNING",e;}}}
function $z(a){a.Qa=!1;if(a.ma)for(var b in a.h)a.h.hasOwnProperty(b)&&a.ma(b,a.h[b]);for(var c in a.P)a.P.hasOwnProperty(c)&&clearTimeout(Number(c));a.P={};a.u=null;a.ma=null;b=a.api;for(var d in b)b.hasOwnProperty(d)&&(b[d]=null);b.addEventListener=function(e,f){a.addEventListener(e,f)};
b.removeEventListener=function(e,f){a.removeEventListener(e,f)};
b.destroy=function(){a.dispose()};
b.getLastError=function(){return a.getLastError()};
b.getPlayerType=function(){return a.getPlayerType()};
b.getCurrentVideoConfig=function(){return a.Ra};
b.loadNewVideoConfig=function(e){a.loadNewVideoConfig(e)};
b.isReady=function(){return a.isReady()}}
p.isReady=function(){return this.Qa};
p.addEventListener=function(a,b){var c=this,d=eA(this,b);d&&(Sb(this.Ic,a)>=0||this.h[a]||(b=pA(this,a),this.u&&this.u(a,b)),this.U.subscribe(a,d),a==="onReady"&&this.isReady()&&setTimeout(function(){d(c.api)},0))};
p.removeEventListener=function(a,b){this.ea||(b=eA(this,b))&&this.U.unsubscribe(a,b)};
function eA(a,b){var c=b;if(typeof b==="string"){if(a.Fa[b])return a.Fa[b];c=function(){var d=C.apply(0,arguments),e=F(b);if(e)try{e.apply(D,d)}catch(f){throw d=new U("PlayerProxy error when executing callback",{error:f}),d.level="ERROR",d;}};
a.Fa[b]=c}return c?c:null}
function pA(a,b){function c(d){function e(){if(!a.ea)try{a.U.sb(b,d!=null?d:void 0)}catch(h){var g=new U("PlayerProxy error when creating global callback",{error:h.message,event:b,playerId:a.A,data:d,originalStack:h.stack,componentStack:h.ge});g.level="WARNING";throw g;}}
if(kA(a,"web_player_publish_events_immediately"))e();else{var f=setTimeout(function(){e();var g=a.P,h=String(f);h in g&&delete g[h]},0);
zg(a.P,String(f))}}
return a.h[b]=c}
p.getPlayerType=function(){return this.Z||(gA(this)?"html5":null)};
p.getLastError=function(){return this.lastError};
function jA(a){a.cancel();$z(a);a.Z=null;a.config&&(a.config.loaded=!1);var b=gA(a);b&&(fA(a)||!mA(a)?a.j=b:(b&&b.destroy&&b.destroy(),a.j=null));if(a.o)for(a=a.o;b=a.firstChild;)a.removeChild(b)}
p.cancel=function(){this.G&&aw(hA(this),this.G);clearTimeout(this.fb);this.Y=!1};
p.ba=function(){jA(this);if(this.j&&this.config&&this.j.destroy)try{this.j.destroy()}catch(b){var a=new U("PlayerProxy error during disposal",{error:b});a.level="ERROR";throw a;}this.Fa=null;for(a in this.h)this.h.hasOwnProperty(a)&&delete this.h[a];this.Ra=this.config=this.api=null;delete this.o;delete this.i;I.prototype.ba.call(this)};
function mA(a){var b,c;a=(b=a.config)==null?void 0:(c=b.args)==null?void 0:c.fflags;return!!a&&a.indexOf("player_destroy_old_version=true")!==-1}
function hA(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.jsUrl:(a=a.config.assets)?a.js:""}
function lA(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.cssUrl:(a=a.config.assets)?a.css:""}
function kA(a,b){if(a.webPlayerContextConfig)var c=a.webPlayerContextConfig.serializedExperimentFlags;else{var d;if((d=a.config)==null?0:d.args)c=a.config.args.fflags}return(c||"").split("&").includes(b+"=true")}
function dA(a){for(var b={},c=y(Object.keys(a)),d=c.next();!d.done;d=c.next()){d=d.value;var e=a[d];b[d]=typeof e==="object"?Cg(e):e}return b}
;var qA={},rA="player_uid_"+(Math.random()*1E9>>>0);function sA(a,b){var c="player",d=!1;d=d===void 0?!0:d;c=typeof c==="string"?Ig(c):c;var e=rA+"_"+Pa(c),f=qA[e];if(f&&d)return tA(a,b)?f.api.loadVideoByPlayerVars(a.args||null):f.loadNewVideoConfig(a),f.api;f=new Zz(c,e,a,b,void 0);qA[e]=f;f.addOnDisposeCallback(function(){delete qA[f.getId()]});
return f.api}
function tA(a,b){return b&&b.serializedExperimentFlags?b.serializedExperimentFlags.includes("web_player_remove_playerproxy=true"):a&&a.args&&a.args.fflags?a.args.fflags.includes("web_player_remove_playerproxy=true"):!1}
;var uA=null,vA=null;
function wA(){Xx();var a=bo(),b=fo(119),c=window.devicePixelRatio>1;if(document.body&&fk(document.body,"exp-invert-logo"))if(c&&!fk(document.body,"inverted-hdpi")){var d=document.body;if(d.classList)d.classList.add("inverted-hdpi");else if(!fk(d,"inverted-hdpi")){var e=dk(d);ek(d,e+(e.length>0?" inverted-hdpi":"inverted-hdpi"))}}else!c&&fk(document.body,"inverted-hdpi")&&gk();if(b!=c){b="f"+(Math.floor(119/31)+1);d=go(b)||0;d=c?d|67108864:d&-67108865;d===0?delete Zn[b]:(c=d.toString(16),Zn[b]=c.toString());
c=!0;T("web_secure_pref_cookie_killswitch")&&(c=!1);b=a.h;d=[];for(f in Zn)Zn.hasOwnProperty(f)&&d.push(f+"="+encodeURIComponent(String(Zn[f])));var f=d.join("&");Vn(b,f,63072E3,a.i,c)}}
function xA(){yA()}
function zA(){Sx("ep_init_pr");yA()}
function yA(){var a=uA.getVideoData(1);a=a.title?a.title+" - YouTube":"YouTube";document.title!==a&&(document.title=a)}
function AA(){uA&&uA.sendAbandonmentPing&&uA.sendAbandonmentPing();R("PL_ATT")&&Wz.dispose();for(var a=Xj,b=0,c=Fy.length;b<c;b++)a.qa(Fy[b]);Fy.length=0;Zv("//static.doubleclick.net/instream/ad_status.js");Gy=!1;Lm("DCLKSTAT",0);yc(vA);uA&&(uA.removeEventListener("onVideoDataChange",xA),uA.destroy())}
;Sx("ep_init_eps");E("yt.setConfig",Lm);E("yt.config.set",Lm);E("yt.setMsg",Wv);E("yt.msgs.set",Wv);E("yt.logging.errors.log",Su);
E("writeEmbed",function(){Sx("ep_init_wes");var a=R("PLAYER_CONFIG");if(!a){var b=R("PLAYER_VARS");b&&(a={args:b})}Jw(!0);a.args.ps==="gvn"&&(document.body.style.backgroundColor="transparent");a.attrs||(a.attrs={width:"100%",height:"100%",id:"video-player"});var c=document.referrer;b=R("POST_MESSAGE_ORIGIN");window!==window.top&&c&&c!==document.URL&&(a.args.loaderUrl=c);c=Vy();if(!c.serializedForcedExperimentIds){var d=Zm(window.location.href);d.forced_experiments&&(c.serializedForcedExperimentIds=
d.forced_experiments)}var e;((e=a.args)==null?0:e.autoplay)?Ox("watch",["pbs","pbu","pbp"]):a.args&&yv(a.args)?Ox("video_preview",["ol"]):Ox("embed_no_video",["ep_init_pr"]);uA=gb(sA(a,c));uA.addEventListener("onVideoDataChange",xA);uA.addEventListener("onReady",zA);a=R("POST_MESSAGE_ID","player");R("ENABLE_JS_API")?vA=new Iz(uA):R("ENABLE_POST_API")&&typeof a==="string"&&typeof b==="string"&&(vA=new Pz(uA,a,b));Hy();T("ytidb_create_logger_embed_killswitch")||ap();a={};tz.h||(tz.h=new tz);tz.h.install((a.flush_logs=
{callback:function(){xu()}},a));
Rs();if(T("embeds_enable_separate_ITS")){Bz();var f=function(){return ay.instance}}else f=function(){var g,h;
if(!Ay){var k=Dt();zt(k,{pb:zy,Hc:xy});var l={Kc:{feedbackEndpoint:Xw(oy),modifyChannelNotificationPreferenceEndpoint:Xw(py),playlistEditEndpoint:Xw(qy),shareEntityEndpoint:Xw(ny),subscribeEndpoint:Xw(ky),unsubscribeEndpoint:Xw(ly),webPlayerShareEntityServiceEndpoint:Xw(ry)}},m=Sw(),n={};m&&(n.client_location=m);g===void 0&&(g=Sn());h===void 0&&(h=k.resolve(zy));by(l,h,g,n);zt(k,{pb:jy,kd:ay.instance});Ay=k.resolve(jy)}return Ay};
T("ytidb_clear_embedded_player")&&Xj.pa(function(){f();gz()});
T("enable_rta_manager")&&po(function(){var g=new Sy(f());var h={preload:!T("enable_rta_npi")},k=!1;if(typeof h==="boolean")var l={preload:h};else typeof h==="undefined"?l={preload:!0}:(l=h,k=!!h.Bh);h=k?void 0:new Bs;Jy.instance=new Jy(g,l,h);g=Jy.instance;l=g.i.bind(g);E("yt.aba.att",l);g=g.j.bind(g);E("yt.aba.att2",g)});
Sx("ep_init_wee")});
E("yt.abuse.player.botguardInitialized",F("yt.abuse.player.botguardInitialized")||Xz);E("yt.abuse.player.invokeBotguard",F("yt.abuse.player.invokeBotguard")||Yz);E("yt.abuse.dclkstatus.checkDclkStatus",F("yt.abuse.dclkstatus.checkDclkStatus")||Iy);E("yt.player.exports.navigate",F("yt.player.exports.navigate")||Iw);E("yt.util.activity.init",F("yt.util.activity.init")||et);E("yt.util.activity.getTimeSinceActive",F("yt.util.activity.getTimeSinceActive")||ht);
E("yt.util.activity.setTimestamp",F("yt.util.activity.setTimestamp")||ft);window.addEventListener("load",Pm(function(){wA()}));
window.addEventListener("pageshow",Pm(function(a){a.persisted||wA()}));
window.addEventListener("pagehide",Pm(function(a){T("embeds_web_enable_dispose_player_if_page_not_cached_killswitch")?AA():a.persisted||AA()}));
window.onerror=function(a,b,c,d,e){var f=d,g;b=b===void 0?"Unknown file":b;c=c===void 0?0:c;d=!1;var h=Mm("log_window_onerror_fraction");if(h&&Math.random()<h)d=!0;else{h=document.getElementsByTagName("script");for(var k=0,l=h.length;k<l;k++)if(h[k].src.indexOf("/debug-")>0){d=!0;break}}if(d){d=!1;e?d=!0:(typeof a==="string"?h=a:ErrorEvent&&a instanceof ErrorEvent?(d=!0,h=a.message,b=a.filename,c=a.lineno,f=a.colno):(h="Unknown error",b="Unknown file",c=0),e=new U(h),e.name="UnhandledWindowError",
e.message=h,e.fileName=b,e.lineNumber=c,isNaN(f)?delete e.columnNumber:e.columnNumber=f);if(!T("wiz_enable_component_stack_propagation_killswitch")){a=e;var m;if((m=g)==null||!m.componentStack)if(a=a.ge){g||(g={});m=g;for(b=[];b.length<20&&a;)b.push(a.name),a=a.parent;a=b.join(" > ");m.componentStack=a}}g&&Wu(e,g);d?Su(e):Tu(e)}};
Ni=Uu;window.addEventListener("unhandledrejection",function(a){Uu(a.reason)});
Tb(R("ERRORS")||[],function(a){Su.apply(null,a)});
Lm("ERRORS",[]);Sx("ep_init_epe");}).call(this);
