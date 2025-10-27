using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    /// <summary>
    /// Provides optional save/load functionality for KelpNet models using DataContractSerializer.
    /// Note: Save/Load is NOT required for normal operation. Models can be created, trained, 
    /// and used without ever calling these methods. This is particularly useful for scenarios
    /// like NinjaTrader indicators where persistence is not desired and models start fresh on each run.
    /// </summary>
    public class ModelIO<T> where T : unmanaged, IComparable<T>
    {
        public static Type[] KnownTypes =
        {
            typeof(NdArray<T>),
            typeof(FunctionDictionary<T>),//Container
            typeof(FunctionStack<T>),
            typeof(DualInputFunction<T>),//Type
            typeof(MultiInputFunction<T>),
            typeof(MultiOutputFunction<T>),
            typeof(SingleInputFunction<T>),
            typeof(SplitFunction<T>),
            typeof(ELU<T>),//Activations
            typeof(Softmax<T>),
            typeof(Swish<T>),
            typeof(Broadcast<T>),//Arrays
            typeof(EmbedID<T>),//Connections
            typeof(LSTM<T>),
            typeof(AddBias<T>),//Mathmetrics
            typeof(MultiplyScale<T>),
            //typeof(StochasticDepth),//Noise
            typeof(BatchNormalization<T>),//Normalization
            typeof(AveragePooling2D<T>),//Poolings
            typeof(LeakyReLU<T>),//Parallizable
            typeof(ReLU<T>),
            typeof(Sigmoid<T>),
            typeof(TanhActivation<T>),
            typeof(Convolution2D<T>),
            typeof(Deconvolution2D<T>),
            typeof(Linear<T>),
            typeof(Dropout<T>),
            typeof(MaxPooling2D<T>)
        };

        /// <summary>
        /// Saves a KelpNet function/model to a file using DataContractSerializer and ZIP compression.
        /// This method is optional - models do not need to be saved to function properly.
        /// </summary>
        /// <param name="function">The function to save</param>
        /// <param name="fileName">The file path where the model will be saved</param>
        public static void Save(Function<T> function, string fileName)
        {
            DataContractSerializer bf = new DataContractSerializer(typeof(Function<T>), new DataContractSerializerSettings { KnownTypes = KnownTypes, PreserveObjectReferences = true });

            //ZIP書庫を作成
            if (File.Exists(fileName))
            {
                File.Delete(fileName);
            }

            using (ZipArchive zipArchive = ZipFile.Open(fileName, ZipArchiveMode.Create))
            {
                ZipArchiveEntry entry = zipArchive.CreateEntry("Function");
                using (Stream stream = entry.Open())
                {
                    bf.WriteObject(stream, function);
                }
            }
        }

        /// <summary>
        /// Loads a previously saved KelpNet function/model from a file.
        /// This method is optional - models can be created and initialized from scratch without loading.
        /// </summary>
        /// <param name="fileName">The file path from which to load the model</param>
        /// <returns>The loaded function</returns>
        public static Function<T> Load(string fileName)
        {
            DataContractSerializer bf = new DataContractSerializer(typeof(Function<T>), new DataContractSerializerSettings { KnownTypes = KnownTypes, PreserveObjectReferences = true });

            using (ZipArchive zipArchive = ZipFile.OpenRead(fileName))
            {
                ZipArchiveEntry zipData = zipArchive.GetEntry("Function");
                return (Function<T>)bf.ReadObject(zipData.Open());
            }
        }
    }
}