var app = new Vue({
    el:'#app',
    data: {
        message: 'Hello Vue!',
        users: [
            {name: 'Adam'},
            {name: 'Ben'},
            {name: 'Chris'}
        ],
        hoge: []
    },
    methods: {
        appendHoge: function(){
            this.hoge.push("hoge!")
        /* 変数へのアクセスはthis.ナントカ. とすればOK */
        }
    }

})