import java.io.File
import java.security.MessageDigest
import javax.inject.Inject

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

/**
 * jetlink's version, from the Python package as the Mac and iPhone apps read
 * it: the release workflow's check ties the tag to the same line.
 */
val jetlinkVersion: String =
    Regex("""__version__ = '([^']+)'""")
        .find(rootDir.resolve("../jetlink/__init__.py").readText())
        ?.groupValues?.get(1) ?: error("no __version__ = '...' in jetlink/__init__.py")

/** 0.5.0 is 500: releases only go up. */
val jetlinkVersionCode: Int =
    jetlinkVersion.split(".").map { it.takeWhile(Char::isDigit).toIntOrNull() ?: 0 }
        .let { (it.getOrElse(0) { 0 } * 10000) + (it.getOrElse(1) { 0 } * 100) + it.getOrElse(2) { 0 } }

android {
    namespace = "io.zoompilot.jetlink"
    compileSdk = 37
    // The NDK the Swift SDK for Android was built with; AGP strips the
    // native libraries with it.
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = "io.zoompilot.jetlink.android"
        // Android 12: Build.SOC_MODEL and the performance hint API.
        minSdk = 31
        targetSdk = 36
        versionCode = jetlinkVersionCode
        versionName = jetlinkVersion
        // The release tag this build came from, for the updater to order with
        // the newest GitHub release. The workflow names it (JETLINK_BUILD_TAG):
        // "v0.8.7" on a tagged release, "0.8.7-1015-1" on a cn-branch build
        // (version-MMDD-that-day's build number), whose push publishes the
        // matching prerelease. GITHUB_REF_NAME is the tap, and "local" a dev build.
        val buildTag = providers.environmentVariable("JETLINK_BUILD_TAG").orNull?.takeIf(String::isNotEmpty)
            ?: providers.environmentVariable("GITHUB_REF_NAME").orNull?.takeIf(String::isNotEmpty)
            ?: "local"
        buildConfigField("String", "BUILD_TAG", "\"$buildTag\"")
        // The QNN runtime is arm64 only, and so is every Snapdragon with an NPU
        // worth driving on. The emulator on an Apple silicon Mac is arm64 too.
        ndk { abiFilters += "arm64-v8a" }
    }

    signingConfigs {
        // The key the APKs on the releases page are signed with, which the
        // Release workflow writes from its secrets (docs/publishing.md).
        // Android installs an update only over an app signed with the same
        // key, so every release has to use this one.
        fun env(name: String) = providers.environmentVariable(name).orNull?.takeIf(String::isNotEmpty)
        env("JETLINK_ANDROID_KEYSTORE")?.let { keystore ->
            create("release") {
                storeFile = file(keystore)
                storePassword = env("JETLINK_ANDROID_KEYSTORE_PASSWORD") ?: error("JETLINK_ANDROID_KEYSTORE_PASSWORD is not set")
                // keytool's PKCS12 keystore has one password, the key's too
                keyPassword = storePassword
                keyAlias = env("JETLINK_ANDROID_KEY_ALIAS") ?: error("JETLINK_ANDROID_KEY_ALIAS is not set")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Sideloaded, never on a store. Without the release key, the debug
            // key: a fork or a local build still builds, and a local release
            // build installs over a debug one.
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        jniLibs {
            // The NPU's skel libraries are loaded by the DSP from a real file,
            // so native libraries are extracted to nativeLibraryDir.
            useLegacyPackaging = true
            // The phone's own QNN set (`src/main/jniLibs`) is the phone's
            // bytes, and the firmware checks them as they are, so nothing
            // here rewrites them on the way into the APK.
            keepDebugSymbols += listOf(
                "**/libQnnHtp.so",
                "**/libQnnHtpPrepare.so",
                "**/libQnnSystem.so",
                "**/libQnnHtpV85Stub.so",
                "**/libQnnHtpV85Skel.so",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

/** The onnxruntime AAR on its own, for its C headers. */
val onnxruntimeAar: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

/**
 * `onnxruntime-android-qnn` with what it depends on: its own AAR, which holds
 * `libonnxruntime.so`, and the `com.qualcomm.qti:qnn-runtime` AAR it names --
 * the backend, its context preparer, its System library, QNN's GPU and DSP
 * backends and the HTP pairs for the Hexagons up to V81. Its Java binding is
 * not wanted: the server calls the C API.
 */
val onnxRuntimeNatives: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = true
}

/**
 * The LiteRT AAR on its own, for its native libraries alone: libLiteRt.so and
 * its GPU accelerator, which the server opens from nativeLibraryDir. Its
 * Kotlin API and what that pulls in (litert-api, Guava, Play's AI delivery)
 * are not wanted.
 */
val litertAar: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

/** LiteRT's NPU runtime libraries, a zip from its GitHub release (settings.gradle.kts). */
val litertNpu: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.service)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    // onnxruntime's native libraries arrive through `onnxRuntimeLibraries`
    // below, which leaves the AAR's own copies of the libraries
    // `src/main/jniLibs` holds out of the APK. The AAR is not an
    // `implementation` dependency for the same reason: the server calls
    // onnxruntime's C API, its Java binding is not wanted, and its telemetry
    // provider is removed from the merge (AndroidManifest.xml).
    debugImplementation(libs.compose.ui.tooling)

    onnxruntimeAar(variantOf(libs.onnxruntime.qnn) { artifactType("aar") })
    onnxRuntimeNatives(libs.onnxruntime.qnn)
    litertAar(variantOf(libs.litert) { artifactType("aar") })
    litertNpu(variantOf(libs.litert.npu) { artifactType("zip") })

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

/**
 * Builds libjetlink.so, the server, with the Swift SDK for Android
 * (scripts/swift-build.sh). Always optimized: the frame path in an
 * unoptimized build is too slow to judge anything by.
 *
 * -Pjetlink.prebuiltSwift=DIR skips the Swift build and packages DIR/arm64-v8a
 * instead, for working on the Kotlin side without the Swift toolchain.
 */
abstract class SwiftBuild @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:OutputDirectory abstract val outputDir: DirectoryProperty
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE) abstract val sources: ConfigurableFileCollection
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val onnxruntime: ConfigurableFileCollection
    @get:Internal abstract val script: RegularFileProperty
    @get:Input @get:Optional abstract val prebuilt: Property<String>
    /** The prebuilt libraries' contents, so a new copy there is picked up. */
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE) abstract val prebuiltFiles: ConfigurableFileCollection

    @TaskAction
    fun build() {
        val out = outputDir.get().asFile
        val prebuiltDir = prebuilt.orNull
        if (prebuiltDir != null) {
            check(File(prebuiltDir, "arm64-v8a/libjetlink.so").isFile) { "$prebuiltDir has no arm64-v8a/libjetlink.so" }
            File(prebuiltDir).copyRecursively(out, overwrite = true)
            return
        }
        exec.exec {
            commandLine("bash", script.get().asFile.absolutePath, out.absolutePath, onnxruntime.singleFile.absolutePath, "release")
        }
    }
}

val swiftBuild = tasks.register<SwiftBuild>("swiftBuild") {
    val kit = rootDir.resolve("../JetlinkKit")
    sources.from(kit.resolve("Package.swift"), fileTree(kit.resolve("Sources")))
    onnxruntime.from(onnxruntimeAar)
    script.set(rootDir.resolve("scripts/swift-build.sh"))
    prebuilt.set(providers.gradleProperty("jetlink.prebuiltSwift"))
    providers.gradleProperty("jetlink.prebuiltSwift").orNull?.let { prebuiltFiles.from(fileTree(it)) }
    outputDir.set(layout.buildDirectory.dir("swift/jniLibs"))
}

/**
 * LiteRT's runtime and GPU accelerator out of its AAR, and Google Tensor's
 * dispatch library and compiler plugin out of the same release's NPU zip,
 * for arm64 as the rest. LiteRT opens them all from nativeLibraryDir. The
 * zip comes from GitHub rather than Maven, so its hash is checked.
 */
abstract class LiteRtLibraries @Inject constructor(
    private val files: FileSystemOperations,
    private val archives: ArchiveOperations,
) : DefaultTask() {
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val aar: ConfigurableFileCollection
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val npu: ConfigurableFileCollection
    @get:Input abstract val npuSha256: Property<String>
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun extract() {
        val zip = npu.singleFile
        val sha256 = MessageDigest.getInstance("SHA-256").digest(zip.readBytes()).joinToString("") { "%02x".format(it) }
        check(sha256 == npuSha256.get()) { "${zip.name} has SHA-256 $sha256, not the ${npuSha256.get()} jetlink was built against" }
        files.sync {
            from(archives.zipTree(aar.singleFile)) {
                include("jni/arm64-v8a/libLiteRt.so", "jni/arm64-v8a/libLiteRtClGlAccelerator.so")
                eachFile { path = "arm64-v8a/$name" }
            }
            from(archives.zipTree(zip)) {
                include(
                    "google_tensor_runtime/src/main/jni/arm64-v8a/libLiteRtDispatch_GoogleTensor.so",
                    "google_tensor_runtime/src/main/jni/arm64-v8a/libLiteRtCompilerPlugin_google_tensor.so",
                )
                eachFile { path = "arm64-v8a/$name" }
                // the zip has them read-only, which the next sync could not replace
                filePermissions { unix("rw-r--r--") }
            }
            includeEmptyDirs = false
            into(outputDir)
        }
    }
}

/**
 * onnxruntime's native libraries, out of its AARs: `libonnxruntime.so` with
 * the QNN provider, QNN's HTP pairs for the Hexagons up to V81, its GPU and
 * DSP backends and its System library, all as the AARs lay them out.
 *
 * Whatever `src/main/jniLibs/arm64-v8a` holds wins over the AAR's copy of the
 * same name, which is how the phone's own QNN set replaces the backend, the
 * context preparer and the System library the AAR ships
 * [docs/qnn-htp-stubs.md]. A checkout without that set gets the AAR's own
 * three, as before.
 */
abstract class OnnxRuntimeLibraries @Inject constructor(
    private val files: FileSystemOperations,
    private val archives: ArchiveOperations,
) : DefaultTask() {
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val aars: ConfigurableFileCollection
    /** The names `src/main/jniLibs/arm64-v8a` holds, which are not taken from the AARs. */
    @get:Input abstract val overrides: ListProperty<String>
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun extract() {
        val ours = overrides.get().map { "jni/arm64-v8a/$it" }
        val trees = aars.files.map { archives.zipTree(it) }
        files.sync {
            from(trees) {
                include("jni/arm64-v8a/*.so")
                if (ours.isNotEmpty()) exclude(*ours.toTypedArray())
                // onnxruntime's Java binding; the server calls the C API.
                exclude("jni/arm64-v8a/libonnxruntime4j_jni.so")
                eachFile { path = "arm64-v8a/$name" }
            }
            includeEmptyDirs = false
            into(outputDir)
        }
    }
}

val onnxRuntimeLibraries = tasks.register<OnnxRuntimeLibraries>("onnxRuntimeLibraries") {
    aars.from(onnxRuntimeNatives)
    overrides.set(file("src/main/jniLibs/arm64-v8a").listFiles()?.map { it.name } ?: emptyList<String>())
    outputDir.set(layout.buildDirectory.dir("onnxruntime/jniLibs"))
}

val liteRtLibraries = tasks.register<LiteRtLibraries>("liteRtLibraries") {
    aar.from(litertAar)
    npu.from(litertNpu)
    // litert_npu_runtime_libraries_jit.zip of LiteRT v2.2.0
    npuSha256.set("d6d160104f110e690f1c9b0c54ab3855af1a12e405dcc6e644dd7e3e1955aac4")
    outputDir.set(layout.buildDirectory.dir("litert/jniLibs"))
}

/**
 * The QNN set this build carries, and what it has to be.
 *
 * `src/main/jniLibs/arm64-v8a` holds the five files of one QNN build: the
 * backend, its context preparer, its System library, and the V85 Stub and
 * Skel for an SM8975, a Snapdragon 8 Elite Gen 6. They are the phone's own,
 * 2.48.5, which is the set this Hexagon opens: the same five of 2.51.40 are
 * refused by the firmware's FastRPC, which asks a Skel for a priority token
 * its build does not have [docs/qnn-htp-stubs.md].
 *
 * Five files of one build, or none of them: half a set is a failure QNN
 * swallows in silence, so the build fails here instead. The hashes are what
 * the build was matched against, and what the running phone reports.
 *
 * A checkout without the set still builds and serves every Hexagon up to V81
 * through the AAR's own libraries (`onnxRuntimeLibraries`), which is what
 * upstream carries; leaving out the V85 pair alone then fails the build, as
 * it did before.
 */
run {
    val phone = linkedMapOf(
        "libQnnHtp.so" to "1108ace00a71059d8b8bde5e63a3c7e296dcdb33e88d789d5e6cc88200a27d59",
        "libQnnHtpPrepare.so" to "8879389024c43226e6c7b441b55146a9d4fd941f931bc2a7bfbc8626645bc02e",
        "libQnnSystem.so" to "8b2f9dffd181bb4753491e78881573a36d5f3ac1af2ca1f61ca24a4e6f2fd9f8",
        "libQnnHtpV85Stub.so" to "3064fd029b4d483fc1365304e4f9a0f06215bef2506d4fa8b6fc6c37209e3880",
        "libQnnHtpV85Skel.so" to "1d4abb21e93817924f8cbcc6535b0b12e14d40276983e5b37caa32744034add9",
    )
    val dir = file("src/main/jniLibs/arm64-v8a")
    val held = phone.keys.filter { File(dir, it).isFile }
    check(held.isEmpty() || held.size == phone.size) {
        "src/main/jniLibs/arm64-v8a holds $held: a QNN set is the five files of one build, " +
            "or none of them"
    }
    if (held.isEmpty()) {
        val v85 = fileTree("src/main/jniLibs/arm64-v8a") { include("libQnnHtpV85*.so") }
        check(v85.files.size == 2) {
            "src/main/jniLibs/arm64-v8a holds ${v85.files.map { it.name }}: the V85 pair is the " +
                "Stub and its Skel together, or nothing at all"
        }
    }
    held.forEach { name ->
        val sha = MessageDigest.getInstance("SHA-256").digest(File(dir, name).readBytes())
            .joinToString("") { "%02x".format(it) }
        check(sha == phone[name]) {
            "$name has SHA-256 $sha, not the ${phone[name]} of the set this build was matched against"
        }
    }
}

androidComponents {
    onVariants { variant ->
        variant.sources.jniLibs?.addGeneratedSourceDirectory(swiftBuild, SwiftBuild::outputDir)
        variant.sources.jniLibs?.addGeneratedSourceDirectory(onnxRuntimeLibraries, OnnxRuntimeLibraries::outputDir)
        variant.sources.jniLibs?.addGeneratedSourceDirectory(liteRtLibraries, LiteRtLibraries::outputDir)
    }
}
