## Description

The input/output of Quantized_MatMul Op is exported here (using node-viewer) for debugging purpose.

## Model Used

reference/models/qntMatMulDeqnt/uint9uint9uint32.py

## Commands

    ipython3
    run reference/models/qntMatMulDeqnt/uint9uint9uint32.py
    import view_node as nv
    nv.init(tf.get_default_graph())
    sess = tf.Session()
    with sess:
        nv.snap("qMatMul")
