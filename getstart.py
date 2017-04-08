import tensorflow as tf

def main():
    # run_const()
    # run_param()
    run_var()


def run_param():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    add_and_triple = adder_node * 3

    sess = tf.Session()
    print(sess.run(add_and_triple, {a:3, b:4.5}))
    print(sess.run(add_and_triple, {a:[1,3], b:[2,4]}))


def run_const():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0,)
    print(node1, node2)

    sess = tf.Session()
    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))


def run_var():
    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    print(sess.run(linear_model, {x:[1,2,3,4]}))


if __name__ == '__main__':
    main()


