// https://www.programmerall.com/article/80951209096/
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

/// Model Prediction Input Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface MLModelInput : NSObject<MLFeatureProvider>

//the input name,default is image
@property (nonatomic, strong) NSString *inputName;

//data as color (kCVPixelFormatType_32BGRA) image buffer
@property (readwrite, nonatomic) MLMultiArray* data;

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithData:(MLMultiArray*)data inputName:(NSString *)inputName;

@end


API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface MLModelOutput : NSObject<MLFeatureProvider>

//the output name, defalut is feature
@property (nonatomic, strong) NSString *outputName;

// feature as multidimensional array of doubles
@property (readwrite, nonatomic) MLMultiArray *feature;

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithFeature:(MLMultiArray *)feature;
@end

NS_ASSUME_NONNULL_END