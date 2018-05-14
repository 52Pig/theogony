
import socket
import redis

r2 = redis.Redis("127.0.0.1", 6380, db=0)

HOST, PORT = '', 8888

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print('Serving HTTP on port %s ...' % PORT)

def Processing(request):
    uid = request.split("key=")[1].split(" ")[0]
    return uid

def gp(uid):
    click_action = {}
    file = open("./logfile.txt")
    for line in file.readlines():
        line = line.strip()
        ls = line.split("&")
        if ls[7] != "1":
            continue
        if ls[1] not in click_action.keys():
            click_action[ls[1]] = []
        click_action[ls[1]].append(ls[4])
    if uid in click_action.keys():
        return "&&".join(click_action[uid])

comment_log = {}

def log_process(request, tag):
    print("here")

def log_process(request, tag):
    # tag=1 文字 返回与文字匹配的商品
    # tag=2 数字 返回与该数字同类目的商品
    if tag == 1:
        if request in cate_its.keys():
            return "&&".join(cate_its[request])


class RTR():
    old = ["1", "2", "3", "4", "5"]
    new = ["5", "4", "3", "2", "1"]
    def p(self, key):
        m = t2.get("9527#1")
        if m != None:
            if int(m) > 5000:
                return "more than 5000"
            else:
                return "less than 5000"
            return ",".join(self.new)
        else:
            return ",".join(self.old)

while True:
    client_conn, client_address = listen_socket.accept()
    request = client_conn.recv(1024)
    paras, tag = request.split("&&")








