import os
from re import template
import tornado.web
from environment import Environment
from planner import ValueIterationPlanner

class IndexHandler(tornado.web.RequestHandler): # RequestHandler を継承するきまり
    
    def get(self): # もともとのgetの書き換え
        self.render("index.html") # index.htmlを書き下す

class Application(tornado.web.Application):
    def __init__(self): # cf. (***)はタプル--書き換え不能--
        handlers = [
            (r"/", IndexHandler),
        ]
        
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            # os.path.dirname(__file__) 実行中のファイルのパス + /template/
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            # 実行中にいじらないデータが 実行中のファイルのパス + /static/ にはいってる
            cookie_secret=os.environ.get("SECRET_TOKEN","__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"),
            # cookieは細かいこと気にしないことにする
            debug=True,
        )
        
        super(Application, self).__init__(handlers, **settings)
