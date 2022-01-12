#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
// #import <Metal/Metal.h>// GPU
// #import <MetalKit/MetalKit.h>//GPU
// #import <CoreImage/CoreImage.h>
#import <CoreML/CoreML.h>
#import "helpers/coreml_helper_feature_provider.h"
// #import "helper_ai.h"
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

MLModel *mlmodel;
NSArray *input_shape;
MLMultiArray * mlinput;
MLMultiArray *mlresult;
MLModelInput *model_input;
MLPredictionOptions *options;
MLModelConfiguration *configuration;
NSURL *model_result_url;
MLModelDescription *mldesc;
NSArray *mlshape;

typedef struct {
   float r;
   float g;
   float b;
} RGBF;

typedef struct {
   uint8_t r;
   uint8_t g;
   uint8_t b;
} RGBU;

 static float clip(float x, float lower, float upper){
   return x < lower ? lower : (x > upper ? upper : x);
    // return 255;
 }//func

static PyObject *
test(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

test2(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyUnicode_FromString("--------->this is test2<----------");
}

//load ml model from file
load(PyObject *self, PyObject *args)
{
    const char* model_path;
    const char* compute_utint;
    bool precompiled=false;//if model is precompiled

    if (!PyArg_ParseTuple(args, "ss", &model_path,&compute_utint)){
        Py_DECREF(NULL);
        return NULL;
    }

    printf(">>load(), params=> compute_unit: %s, model_path: %s\n",compute_utint,model_path);
    NSError *error=nil;
    NSString *_model_path=[NSString stringWithUTF8String:model_path];

    if([[_model_path pathExtension] isEqualToString:@"mlmodelc"]){
        precompiled=true;
    }

    NSURL *model_result_url=[NSURL fileURLWithPath:_model_path];//isDirectory:FALSE relativeToURL:nil 
    if(!precompiled){
        model_result_url   =[MLModel compileModelAtURL:model_result_url error: &error];
    }
    if(error!=NULL){
        NSLog(@"-->faied to compile ml model, error: %@",error);
        Py_DECREF(NULL);
        return NULL;
    }

    configuration= [MLModelConfiguration new];
    configuration.computeUnits = MLComputeUnitsAll;//MLComputeUnitsAll MLComputeUnitsCPUOnly MLComputeUnitsCPUAndGPU
    if(strcmp(compute_utint,"CPU")==0){configuration.computeUnits = MLComputeUnitsCPUOnly;}
        else if(strcmp(compute_utint,"GPU")==0){configuration.computeUnits = MLComputeUnitsCPUAndGPU;}
        else {configuration.computeUnits = MLComputeUnitsAll;}

    mlmodel=[MLModel modelWithContentsOfURL:model_result_url configuration:configuration  error: &error];
    if(error!=NULL){
        NSLog(@"-->init  ml model, error: %@",error);
        Py_DECREF(NULL);
        return NULL;
    }

    mldesc=mlmodel.modelDescription;
    NSLog(@"mlmodel: %@",mldesc);
    mlshape=mldesc.inputDescriptionsByName[@"input"].multiArrayConstraint.shape;//@[@1,@3,@256,@256];//FIXME: getit from model

    options=[[MLPredictionOptions alloc] init];
    mlinput=[[MLMultiArray alloc] initWithShape:mlshape dataType:mldesc.inputDescriptionsByName[@"input"].multiArrayConstraint.dataType error:&error];
    model_input = [[MLModelInput alloc] initWithData:mlinput inputName:@"input"];
    if(error!=NULL){
        NSLog(@"-->failed to setup mlinput, error: %@",error);
        Py_DECREF(NULL);
        return NULL;
    }

    return PyUnicode_FromString(model_path);//return back model path
}//load

//predict
predict(PyObject *self, PyObject *args)
{
    PyObject *arg1=NULL;
    PyArrayObject *arr1=NULL;
    PyArrayObject *arr_out=NULL;
    PyObject *clip_min=NULL;
    PyObject *clip_max=NULL;


    if (!PyArg_ParseTuple(args, "OOO", &arg1,&clip_min,&clip_max)){
        Py_DECREF(NULL);
        return NULL;
    }

    // arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT32, NPY_ARRAY_ENSUREARRAY);//NPY_ARRAY_IN_ARRAY
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT32,NPY_ARRAY_C_CONTIGUOUS);//NPY_ARRAY_IN_ARRAY

    if (arr1 == NULL){
        printf("==>Cant parse input arr, is null.\n");
        Py_DECREF(NULL);
        return NULL;
    }

    npy_intp *shape=PyArray_SHAPE(arr1);
    // printf(">>>dims: %d, type: %d, shape: (%d,%d,%d,%d)\n",PyArray_NDIM(arr1),PyArray_TYPE(arr1),shape[0],shape[1],shape[2],shape[3]);

    //FIXME: try to prevent copy to improve performance
    //FIXME: add clip and norm 0..255 (native not numpy)
    //FIXME: add cast to uint8, check memory leaks
    NSError *error=nil;
    memcpy((float*)mlinput.dataPointer,(float*)PyArray_DATA(arr1),sizeof(float)*PyArray_SIZE(arr1));//shape[0]*shape[1]*shape[2]*shape[3]
    id<MLFeatureProvider> model_output = [mlmodel predictionFromFeatures:model_input options:options error:&error];
    if(error!=NULL){
        NSLog(@"-->failed to predict, mlmodel error: %@",error);
        Py_DECREF(NULL);
        return NULL;
    }
    mlresult = [model_output featureValueForName:@"output"].multiArrayValue;
    // printf("-->shape: %d,%d,%d,%d\n",(int)[(NSNumber *)[mlresult.shape objectAtIndex:0] integerValue],(int)[(NSNumber *)[mlresult.shape objectAtIndex:1] integerValue],(int)[(NSNumber *)[mlresult.shape objectAtIndex:2] integerValue],(int)[(NSNumber *)[mlresult.shape objectAtIndex:3] integerValue]);
    int nd = 4;
    npy_intp dims[4] = {(int)[(NSNumber *)[mlresult.shape objectAtIndex:0] integerValue],(int)[(NSNumber *)[mlresult.shape objectAtIndex:1] integerValue],(int)[(NSNumber *)[mlresult.shape objectAtIndex:2] integerValue],(int)[(NSNumber *)[mlresult.shape objectAtIndex:3] integerValue]};
    arr_out=PyArray_SimpleNewFromData(nd,dims,NPY_FLOAT,mlresult.dataPointer);//mlresult.dataPointer

    //try transpose of array
    PyArray_Dims newaxes;
    npy_intp _dims[3]={1,2,0};
    newaxes.ptr=_dims;
    newaxes.len=3;

    arr_out=(PyArrayObject*)PyArray_Squeeze(arr_out);
    arr_out=(PyArrayObject*)PyArray_Transpose(arr_out,&newaxes);
    // PyObject *min=PyLong_FromDouble(0.0f);
    // PyObject *max=PyLong_FromDouble(1.0f);
    // arr_out=(PyArrayObject*)PyArray_Clip(arr_out,clip_min,clip_max,false);
    // arr_out=(PyArrayObject*)PyArray_Cast(arr_out,NPY_INT16);
    
    // Convert result arr from model to uit8_t with clipping to 0..255
    //get c-style array
    PyArrayObject *arr_out_copy=(PyArrayObject*)PyArray_FromArray(arr_out, NULL, NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ENSURECOPY|NPY_ARRAY_C_CONTIGUOUS);
    // PyArray_AsCArray(arr_out, void* ptr, npy_intp* dims, int nd, int typenum, int itemsize)
    float* arr_src=(float*)PyArray_DATA(arr_out_copy);//PyArray_DATA PyArray_BYTES
    npy_intp *out_shape=PyArray_SHAPE(arr_out_copy);
    int height=out_shape[0];
    int width=out_shape[1];
    int channels=out_shape[2];
    int linesize=width*channels;//w*c
    uint8_t *buff_out=malloc(sizeof(uint8_t)*width*height*channels);
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
                     RGBF rgbf = *((RGBF *)(arr_src + y * linesize) + x);
                    // printf(">>rgb: (%.1f,%.1f,%.1f)",rgbf.r,rgbf.g,rgbf.b);
                    // printf("--> (%.1f,%.1f,%.1f",arr_src[y*linesize+x][0],arr_src[y*linesize+x][1],arr_src[y*linesize+x][2]);
                    //uit8
                    RGBU rgbu;
                    rgbu.r=(uint8_t)clip(rgbf.r*255.0f,0.0f,255.0f);
                    rgbu.g=(uint8_t)clip(rgbf.g*255.0f,0.0f,255.0f);
                    rgbu.b=(uint8_t)clip(rgbf.b*255.0f,0.0f,255.0f);
                    *((RGBU *)(buff_out + y * linesize) + x)=rgbu;

                    //float
                    // RGBF rgbf2;
                    // rgbf.r=255.0f;//(float)clip(rgbf.r*255.0f,0.0f,255.0f);
                    // rgbf.g=255.0f;//(float)clip(rgbf.g*255.0f,0.0f,255.0f);
                    // rgbf.b=255.0f;//(float)clip(rgbf.b*255.0f,0.0f,255.0f);
                    // *((RGBF *)(buff_out + y * linesize) + x)=rgbf2;
                    // memcpy(buff_out+y*linesize,arr_src+y*linesize,sizeof(float)*linesize);

        }//for x
    }//for y

    npy_intp __dims[3]={height,width,channels};
    PyArrayObject *arr_out_uint8=PyArray_SimpleNewFromData(3, __dims, NPY_UINT8, (uint8_t*)buff_out);
    PyArray_ENABLEFLAGS(arr_out_uint8, NPY_ARRAY_OWNDATA);//18ms
    return arr_out_uint8;
    // return arr_out;
}//predict

static PyMethodDef MyCoremlMethods[] = {
    {"test",  test, METH_VARARGS,"Execute a shell command."},
    {"test2",  test2, METH_VARARGS,"Execute a shell command 2."},
    {"load",  load, METH_VARARGS,"Load model from file"},
    {"predict",  predict, METH_VARARGS,"Predict on model"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef mycoremlmodule = {
    PyModuleDef_HEAD_INIT,
    "mycoreml",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    MyCoremlMethods
};

PyMODINIT_FUNC
PyInit_mycoreml(void)
{
    import_array();
    return PyModule_Create(&mycoremlmodule);
}