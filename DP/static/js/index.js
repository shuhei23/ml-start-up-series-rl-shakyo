Vue.config.debug = true;
Vue.config.devtools = true;
var app = new Vue({
    el:'#app',
    data: {
        flag: true,
        users: [
            {name: 'Adam'},
            {name: 'Ben'},
            {name: 'Crhis'}
        ],
        hello: []
    },
    methods: {
        appendHelllo: function(){
            this.hello.push("hello");
        }
    }
})