#import "coreml_helper_feature_provider.h"


@implementation MLModelInput

- (instancetype)initWithData:(MLMultiArray*)data inputName:(nonnull NSString *)inputName {
    if (self) {
        _data = data;
        _inputName = inputName;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[self.inputName]];
}

- (nullable MLFeatureValue *)featureValueForName:(nonnull NSString *)featureName {
    if ([featureName isEqualToString:self.inputName]) {
        return [MLFeatureValue featureValueWithMultiArray:_data];
    }
    return nil;
}

@end


@implementation MLModelOutput

- (instancetype)initWithFeature:(MLMultiArray *)feature{
    if (self) {
        _feature = feature;
        _outputName = @"output";
    }
    return self;
}

- (NSSet<NSString *> *)featureNames{
    return [NSSet setWithArray:@[self.outputName]];
}

- (nullable MLFeatureValue *)featureValueForName:(nonnull NSString *)featureName {
    if ([featureName isEqualToString:self.outputName]) {
        return [MLFeatureValue featureValueWithMultiArray:_feature];
    }
    return nil;
}


@end