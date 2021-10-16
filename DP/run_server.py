import os
import tornado.ioloop
from tornado.options import define, options, parse_command_line
from application import Application

define("port", default=8888, help="run on the given port", type=int)

def main():
    parse_command_line()
    # https://conta.hatenablog.com/entry/2012/05/31/222940
    # 上の記事の「全体像＋mainのところ」を読むとparse_command_line()の説明がある
    # python run_server.py --8080 とかやるとポートを指定できるようなコマンドラインのオプションが作れる
    app = Application()
    port = int(os.environ.get("PORT", 8888)) 
    app.listen(port) # Application が listenしている...
    print("Run server on port:{}".format(port))
    tornado.ioloop.IOLoop.current().start()
    
if __name__ == "__main__":
    main()