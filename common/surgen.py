import onnx
import numpy as np
import onnx_graphsurgeon as gs
def surgen():
    print('start layout surgen ...\n')
    model = onnx.load("./onnx/layout.onnx")
    graph = gs.import_onnx(model)
    num = 0
    for node in graph.nodes:
        if node.name == "Cast_113":
            outTensor = gs.Variable(name="tensor", dtype=np.int32, shape=None)
            castNode = gs.Node(name="Cast_mm", op="Cast", attrs={"to": np.int32}, inputs=node.outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o().inputs.index(node.outputs[0])
            node.o().inputs[index] = outTensor


        if node.name == "Clip_51":
            outTensor = gs.Variable(name="tensor2", dtype=np.int32, shape=None)
            castNode = gs.Node(name="Cast_mnn", op="Cast", attrs={"to": np.int32}, inputs=node.outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o().inputs.index(node.outputs[0])
            node.o().inputs[index] = outTensor


        if node.name == "Clip_44":
            outTensor = gs.Variable(name="tensor4", dtype=np.int32, shape=None)
            castNode = gs.Node(name="Cast_mnnn", op="Cast", attrs={"to": np.int32}, inputs=node.outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o().inputs.index(node.outputs[0])
            node.o().inputs[index] = outTensor

        if node.name == "Sub_50":
            outTensor = gs.Variable(name="tensor1", dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_x", op="Cast", attrs={"to": np.float32}, inputs=node.outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o().inputs.index(node.outputs[0])
            node.o().inputs[index] = outTensor

            node.o().o().inputs[1] = gs.Constant(name="x", values=np.array([0], dtype=np.float32))
            node.o().o().inputs[2] = gs.Constant(name="y", values=np.array([1032], dtype=np.float32))
      
        if node.name == "Sub_43":
            outTensor = gs.Variable(name="tensor3", dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_y", op="Cast", attrs={"to": np.float32}, inputs=node.outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o().inputs.index(node.outputs[0])
            node.o().inputs[index] = outTensor

            node.o().o().inputs[1] = gs.Constant(name="x1", values=np.array([0], dtype=np.float32))
            node.o().o().inputs[2] = gs.Constant(name="y1", values=np.array([1032], dtype=np.float32))


        if node.op == 'Sub' and node.o(0).op == 'Cast' and node.o(0).o().op == 'Greater':# and node.o(1).op == 'Abs':# and node.o(1).o(1).op == 'Less': #and \
            # Abs
            outTensor = gs.Variable(name="tensor6"+str(num), dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_s"+str(num), op="Cast", attrs={"to": np.float32}, inputs=node.o(1).outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o(1).o(2,0).inputs.index(node.o(1).outputs[0])
            node.o(1).o(2,0).inputs[index] = outTensor
            # constant
            outTensor = gs.Variable(name="tensor186"+str(num), dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_q"+str(num), op="Cast", attrs={"to": np.float32}, inputs=node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().outputs, outputs= [outTensor])
            graph.nodes.append(castNode)

            index = node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().o().inputs.index(
                node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().outputs[0])
            node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().o().inputs[index] = outTensor
            # min_add
            outTensor = gs.Variable(name="tensor166" + str(num), dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_e" + str(num), op="Cast", attrs={"to": np.float32},
                            inputs=node.o(1).o(1).o().o().o().o().o().o().o().o().o().outputs, outputs=[outTensor])
            graph.nodes.append(castNode)
            print(node.o(1).o(1).o().o().o().o().o().o().o().o().o().o(1).name)
            index = node.o(1).o(1).o().o().o().o().o().o().o().o().o().o(1).inputs.index(node.o(1).o(1).o().o().o().o().o().o().o().o().o().outputs[0])
            node.o(1).o(1).o().o().o().o().o().o().o().o().o().o(1).inputs[index] = outTensor

            num += 1

        if node.op == 'Sub' and node.o(0).op == 'Greater' and node.o(0).o().op == 'Cast':

            # Abs
            outTensor = gs.Variable(name="tensor6"+str(num), dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_t"+str(num), op="Cast", attrs={"to": np.float32}, inputs=node.o(1).outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o(1).o(2,0).inputs.index(node.o(1).outputs[0])
            node.o(1).o(2,0).inputs[index] = outTensor
            # constant
            outTensor = gs.Variable(name="tensor186" + str(num), dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_b" + str(num), op="Cast", attrs={"to": np.float32},
                            inputs=node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().outputs,
                            outputs=[outTensor])
            graph.nodes.append(castNode)

            index = node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().o().inputs.index(
                node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().outputs[0])
            node.o(1).o(1).o().o().o().o().o().o().o().o().o().o().o().o().inputs[index] = outTensor
            # min_add
            outTensor = gs.Variable(name="tensor166" + str(num), dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_m" + str(num), op="Cast", attrs={"to": np.float32},
                            inputs=node.o(1).o(1).o().o().o().o().o().o().o().o().o().outputs, outputs=[outTensor])
            graph.nodes.append(castNode)
            index = node.o(1).o(1).o().o().o().o().o().o().o().o().o().o(1).inputs.index(
                node.o(1).o(1).o().o().o().o().o().o().o().o().o().outputs[0])
            node.o(1).o(1).o().o().o().o().o().o().o().o().o().o(1).inputs[index] = outTensor

            # add_add
            outTensor = gs.Variable(name="tensor11", dtype=np.float32, shape=None)
            castNode = gs.Node(name="Cast_y11", op="Cast", attrs={"to": np.float32}, inputs=node.o(0).o().o().o().outputs, outputs= [outTensor])
            graph.nodes.append(castNode)
            index = node.o(0).o().o().o().o().inputs.index(node.o(0).o().o().o().outputs[0])
            node.o(0).o().o().o().o().inputs[index] = outTensor
            num += 1

        # OneHot Cast MatMul convert to Gather

        if node.op == "OneHot":
            indices = gs.Variable(name="tensor" + node.name, dtype=np.int32, shape=None)
            cast_node = gs.Node(name="Cast" + node.name, op="Cast", attrs={"to": np.int32}, inputs=node.i().outputs,
                                outputs=[indices])
            graph.nodes.append(cast_node)
            data = node.o().o().inputs[1]
            gather_out = gs.Variable(name="cus_gather_out" + node.name, dtype=np.float32, shape=None)
            gather_node = gs.Node(name="cus_gather" + node.name, op="Gather", inputs=[data, indices],
                                outputs=node.o().o().outputs)
            graph.nodes.append(gather_node)
            node.inputs = []
            node.o().o().outputs = []

    model = onnx.load('./onnx/surgen.onnx')
    out0_dim_proto_1 = model.graph.output[0].type.tensor_type.shape.dim[1]
    out0_dim_proto_1.dim_param = '709'
    onnx.save(model, './onnx/surgen.onnx')
    print('layout surgen success\n')


def layernorm_surgen():
    model = onnx.load("./onnx/surgen.onnx")
    graph = gs.import_onnx(model)
    for node in graph.nodes: 
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.name not in ['ReduceMean_55'] and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):
                
            data = node.inputs[0]
            gama = node.o().o(1, 0).o().inputs[1]
            beta = node.o().o(1, 0).o().o().inputs[1]
            ln_res = node.o().o(1,0).o().o().outputs[0]
            layer_norm = gs.Node(name = 'layer_norm_' + node.name, op = 'LayerNorm', inputs = [data, gama, beta], outputs= [ln_res])
            # Add_301 outputs
            #add_outputs = node.o(1,0).o(1,0).o().o().outputs[0].outputs
            
            '''
            assert len(add_outputs) == 4, add_outputs[0].name
            
            print(add_outputs)
            print(type(add_outputs))
            print(add_outputs[2])
            
            # SynchronizedList can not use for like this
            # for i in add_outputs:

            for i in range(len(add_outputs)):
                print(i)
                ele = add_outputs[i]
                # change Add_301 4 outputs to ln_res 
                if ele.op != 'Add':
                    ele.inputs[0] = ln_res
                else:
                    ele.inputs[1] = ln_res
                    # cut Add_30
                    '''
            graph.nodes.append(layer_norm)
            node.o().o(1,0).o().o().outputs = []

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "./onnx/layernorm_plugin.onnx")


